"""Identity of ONE ``complete_task`` attempt, and the receipt it left behind.

Several attempts on the same card are ordinary: a worker retries, an operator
completes by hand, a dispatcher and a human race for the same gate. Both opt-in
completion gates persist their receipt in the very transaction that refuses the
transition, so "why did MY attempt fail?" can only be answered by reading back
the receipt THIS attempt wrote.

An event-id floor alone cannot do that. The floor is read before the attempt
starts and the gates verify GitHub/git with no transaction open, so a receipt
another connection writes meanwhile lands above the floor too — and reporting
it tells the operator their completion failed for a condition some other
attempt hit. Each attempt therefore carries its own opaque id, stamped onto
every receipt it persists, and a reader matches that id exactly. The floor
survives only as a cheap bound on how far back the lookup has to scan.
"""
from __future__ import annotations

import uuid

#: Receipt kinds the two opt-in completion gates append.
COMPLETION_GATE_KINDS = ("pr_acceptance", "integration_acceptance")
#: The receipt field carrying the attempt that wrote it.
ATTEMPT_ID_KEY = "completion_attempt_id"


def new_completion_attempt_id() -> str:
    """An id for one completion attempt.

    Opaque on purpose: it is only ever compared for equality, never parsed,
    ordered or shown, so nothing can come to depend on its shape.
    """
    return uuid.uuid4().hex


def stamp_attempt(receipt: dict, attempt_id: str | None) -> dict:
    """Record which attempt a receipt belongs to, returning that receipt."""
    if isinstance(receipt, dict) and attempt_id:
        receipt[ATTEMPT_ID_KEY] = attempt_id
    return receipt


def refusal_receipt(conn, task_id: str, attempt_id: str | None, event_floor: int):
    """``(kind, receipt)`` of the gate receipt ``attempt_id`` persisted, else None.

    Only an exact id match counts, so a receipt from a concurrent attempt — or
    one left behind by an earlier attempt below the floor — is never reported as
    this attempt's reason.
    """
    if not attempt_id:
        return None
    from hermes_cli.kanban_db import _json_dict

    for row in conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND id > ? "
        f"AND kind IN ({', '.join('?' * len(COMPLETION_GATE_KINDS))}) ORDER BY id DESC",
        (task_id, event_floor, *COMPLETION_GATE_KINDS),
    ):
        receipt = _json_dict(row["payload"])
        if receipt.get(ATTEMPT_ID_KEY) == attempt_id:
            return row["kind"], receipt
    return None
