"""Integration-gate declarations and the gate half of the terminal write.

Owns the ``integration_gates`` rows and the snapshot discipline around
:func:`~hermes_cli.kanban_integration_gate.collect_gate_acceptance`: the board
facts the verification was computed from are captured first, the network and
git work happens with no transaction open, and the snapshot is rechecked
inside ``complete_task``'s write transaction before the terminal UPDATE. A
run/status/config change in between rejects the attempt rather than promoting
the gate's child on stale evidence.

``complete_task`` stays the terminal transition owner; this module only
answers "may it?" and records the receipt.
"""
from __future__ import annotations

import json
import time
from dataclasses import astuple
from pathlib import Path
from typing import Optional

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_integration_gate import GateConfig, collect_gate_acceptance

_GATE_COLUMNS = (
    "gate_task_id, implementation_task_id, qa_task_id, repository_path, "
    "integration_remote, integration_branch, require_human_merge"
)
#: Statuses ``complete_task`` accepts as a source for its terminal UPDATE.
_COMPLETABLE = {"running", "ready", "blocked", "review"}


class IntegrationGateConfigError(ValueError):
    """A gate declaration the board cannot honour (bad ids, graph or paths)."""


# --- declarations ---

def get_gate(conn, gate_task_id: str) -> Optional[GateConfig]:
    """The gate declared for ``gate_task_id``, or None when the card is an
    ordinary one. An empty ``integration_gates`` table makes every card
    ordinary, which is what keeps the feature opt-in."""
    row = conn.execute(
        f"SELECT {_GATE_COLUMNS} FROM integration_gates WHERE gate_task_id = ?", (gate_task_id,),
    ).fetchone()
    return _row_to_config(row) if row is not None else None


def list_gates(conn) -> list[GateConfig]:
    return [_row_to_config(row) for row in conn.execute(
        f"SELECT {_GATE_COLUMNS} FROM integration_gates ORDER BY gate_task_id")]


def _row_to_config(row) -> GateConfig:
    return GateConfig(
        gate_task_id=row["gate_task_id"],
        implementation_task_id=row["implementation_task_id"],
        qa_task_id=row["qa_task_id"],
        repository_path=row["repository_path"],
        integration_remote=row["integration_remote"],
        integration_branch=row["integration_branch"],
        require_human_merge=bool(row["require_human_merge"]),
    )


def configure_gate(
    conn, gate_task_id: str, *, implementation_task_id: str, qa_task_id: str,
    repository_path: str, integration_remote: str = "origin",
    integration_branch: str = "develop", require_human_merge: bool = True,
) -> GateConfig:
    """Declare (or re-declare) the gate on ``gate_task_id``.

    Implementation and QA must ALREADY be direct parents of the gate card:
    declaring a gate is not allowed to rewrite an existing board's graph
    implicitly, so a missing edge is an error naming the ``hermes kanban link``
    that fixes it. Only the declaration row and an audit event are written —
    no card changes status, assignee or links.
    """
    gate_task_id = _require_id(gate_task_id, "gate task id")
    implementation_task_id = _require_id(implementation_task_id, "implementation task id")
    qa_task_id = _require_id(qa_task_id, "QA task id")
    if len({gate_task_id, implementation_task_id, qa_task_id}) != 3:
        raise IntegrationGateConfigError(
            "gate, implementation and QA must be three different tasks")
    remote = _require_ref_token(integration_remote, "integration remote")
    branch = _require_ref_token(integration_branch, "integration branch")
    repo_path = _validated_repository_path(repository_path)
    from hermes_cli.kanban_db import _append_event

    with write_txn(conn):
        for label, task_id in (("gate", gate_task_id),
                               ("implementation", implementation_task_id),
                               ("QA", qa_task_id)):
            if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
                raise IntegrationGateConfigError(f"unknown {label} task {task_id}")
        for label, parent_id in (("implementation", implementation_task_id), ("QA", qa_task_id)):
            if conn.execute(
                "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, gate_task_id),
            ).fetchone() is None:
                raise IntegrationGateConfigError(
                    f"{label} task {parent_id} is not a direct parent of gate {gate_task_id}; "
                    f"link it first (`hermes kanban link {parent_id} {gate_task_id}`). "
                    f"Configuring a gate never rewrites the board graph."
                )
        conn.execute(
            f"INSERT INTO integration_gates ({_GATE_COLUMNS}, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(gate_task_id) DO UPDATE SET "
            "implementation_task_id = excluded.implementation_task_id, "
            "qa_task_id = excluded.qa_task_id, repository_path = excluded.repository_path, "
            "integration_remote = excluded.integration_remote, "
            "integration_branch = excluded.integration_branch, "
            "require_human_merge = excluded.require_human_merge",
            (gate_task_id, implementation_task_id, qa_task_id, repo_path, remote, branch,
             1 if require_human_merge else 0, int(time.time())),
        )
        config = get_gate(conn, gate_task_id)
        _append_event(conn, gate_task_id, "integration_gate_configured", config.as_dict())
    return config


def remove_gate(conn, gate_task_id: str) -> bool:
    """Drop the declaration, returning the card to ordinary parent gating."""
    from hermes_cli.kanban_db import _append_event

    with write_txn(conn):
        removed = conn.execute(
            "DELETE FROM integration_gates WHERE gate_task_id = ?", (gate_task_id,)).rowcount == 1
        if removed:
            _append_event(conn, gate_task_id, "integration_gate_removed", None)
    return removed


def _require_id(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationGateConfigError(f"{label} is required")
    return value.strip()


def _require_ref_token(value, label: str) -> str:
    """Remote/branch names are interpolated into a git refspec, so reject
    anything that is not a plain name."""
    token = value.strip() if isinstance(value, str) else ""
    if not token or token.startswith("-") or any(c.isspace() for c in token) or ".." in token \
            or any(c in token for c in "~^:?*[\\"):
        raise IntegrationGateConfigError(f"{label} must be a plain git ref name, got {value!r}")
    return token


def _validated_repository_path(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationGateConfigError("repository path is required")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise IntegrationGateConfigError(
            f"repository path must be absolute, got {value!r} (the gate is verified by whichever "
            f"process completes it, not from your shell's cwd)")
    if not path.is_dir():
        raise IntegrationGateConfigError(f"repository path {path} is not a directory")
    return str(path)


# --- terminal-transition gate ---

def prepare_integration_gate(conn, task_id: str, expected_run_id: Optional[int]):
    """``None`` when the card is not a declared gate, ``False`` when the
    caller's run/status snapshot is already lost, else ``(snapshot, receipt)``.

    The external verification runs here — deliberately BEFORE
    ``complete_task`` opens its write transaction, so no network or git call
    ever holds the board's write lock.
    """
    config = get_gate(conn, task_id)
    if config is None:
        return None
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status = snapshot[0], snapshot[1]
    if status not in _COMPLETABLE or (expected_run_id is not None and run_id != expected_run_id):
        return False
    return snapshot, collect_gate_acceptance(config, _board_evidence(conn, config))


def record_integration_gate(conn, task_id: str, prepared) -> bool:
    """Called under ``complete_task``'s write_txn, before its terminal UPDATE.

    Persists the immutable receipt either way: a failed gate must leave
    diagnostics behind without making the gate — or its child — executable.
    """
    from hermes_cli.kanban_db import _append_event

    snapshot, receipt = prepared
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "integration_acceptance", receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            (f"Integration gate {receipt['phase']}: {receipt.get('detail', '')} "
             f"{receipt['recovery']}", task_id),
        )
    return bool(receipt["ok"])


def _snapshot(conn, task_id: str):
    """The gate's run/status, its declaration, and the parent facts the receipt
    was computed from — the tuple rechecked under the terminal write's lock."""
    row = conn.execute(
        "SELECT current_run_id, status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    config = get_gate(conn, task_id)
    if row is None or config is None:
        return None
    parents = tuple(
        (conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (parent_id,))
         .fetchone() or {"status": None, "current_run_id": None})
        for parent_id in (config.implementation_task_id, config.qa_task_id)
    )
    return (
        row["current_run_id"], row["status"], astuple(config),
        tuple((p["status"], p["current_run_id"]) for p in parents),
        _latest_completed_qa_run_id(conn, config.qa_task_id),
    )


def _latest_completed_qa_run_id(conn, qa_task_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
        "ORDER BY id DESC LIMIT 1", (qa_task_id,),
    ).fetchone()
    return int(row["id"]) if row is not None else None


def _board_evidence(conn, config: GateConfig) -> dict:
    """Everything the verification needs from the board, read in one pass."""
    impl = conn.execute(
        "SELECT status, completion_contract FROM tasks WHERE id = ?",
        (config.implementation_task_id,)).fetchone()
    qa = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (config.qa_task_id,)).fetchone()
    qa_run = conn.execute(
        "SELECT id, summary, metadata FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
        "ORDER BY id DESC LIMIT 1", (config.qa_task_id,)).fetchone()
    qa_metadata = _json_dict(qa_run["metadata"]) if qa_run is not None else {}
    accepted = _accepted_acceptance_receipt(conn, config.implementation_task_id)
    return {
        "implementation_status": impl["status"] if impl is not None else None,
        "contract": impl["completion_contract"] if impl is not None else None,
        "qa_status": qa["status"] if qa is not None else None,
        "qa_run_id": int(qa_run["id"]) if qa_run is not None else None,
        "qa_decision": _str_or_none(qa_metadata.get("decision")),
        "qa_revision": _str_or_none(qa_metadata.get("revision")),
        "qa_summary_mentions_pass": "pass" in str(
            (qa_run["summary"] if qa_run is not None else "") or "").lower(),
        "accepted_head_sha": accepted.get("head_sha"),
        "pr_url": accepted.get("pr_url"),
    }


def _accepted_acceptance_receipt(conn, implementation_task_id: str) -> dict:
    """The newest ACCEPTED ``pr_acceptance`` receipt on the implementation card.

    That receipt is the only durable record of which exact head GitHub
    acceptance actually passed, which is the head QA's revision must match.
    """
    for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'pr_acceptance' "
        "ORDER BY id DESC", (implementation_task_id,),
    ):
        payload = _json_dict(row["payload"])
        if payload.get("ok"):
            return payload
    return {}


def _json_dict(raw) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _str_or_none(value):
    return value if isinstance(value, str) and value.strip() else None
