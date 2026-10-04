"""Behavioural invariants for the opt-in integration gate.

A gate card sits below an Implementation card and its QA card and above the
next Implementation card. Completing it must prove the one thing the board
cannot see by itself: the implementation whose QA passed is *actually
integrated* — the exact PR merged (by a human) into the configured integration
branch, with that merge commit present in the freshly fetched branch.

Everything external is real except GitHub: a temporary bare origin and a
temporary clone exercise the real ``git fetch`` / ``git merge-base
--is-ancestor`` the gate relies on (squash merges mean only
``merge_commit_sha`` is ever in the branch), real SQLite and the real
``complete_task`` lifecycle run, and GitHub is mocked at
``kanban_pr_acceptance._api`` — the single process boundary both halves of the
completion gate reach GitHub through.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd
from hermes_cli import kanban_integration_gate_store as gates
from hermes_cli import kanban_pr_acceptance as pra
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_integration_gate import UNPROVABLE_PHASES, describe_receipt

PR_URL = "https://github.com/acme/repo/pull/7"
HEAD = "a" * 40
OTHER_HEAD = "b" * 40
#: A well-formed sha that names no object in the gate's clone.
UNKNOWN_SHA = "c" * 40


# --------------------------------------------------------------------------
# Mocked GitHub
# --------------------------------------------------------------------------

class FakeGitHub:
    """``gh api`` responses in the shapes GitHub actually returns.

    Defaults describe the repository the mandate targets: a private repo on a
    free plan, so the Repository Rules endpoint answers 403 and the required
    checks come from the declared policy.
    """

    def __init__(self):
        self.head = HEAD
        self.base = "develop"
        self.state = "OPEN"
        self.merged = False
        self.merge_commit_sha = None
        self.merged_by = {"login": "maintainer", "type": "User"}
        self.check_runs = []
        self.rules_forbidden = True
        self.pull_read_fails = False
        self.calls = []
        self.hooks = {}

    def green(self, name="ci/test"):
        self.check_runs.append({
            "id": 42, "name": name, "head_sha": self.head, "app": {"id": 1},
            "status": "completed", "conclusion": "success",
            "html_url": "https://github.com/acme/repo/actions/runs/42",
        })
        return self

    def merge(self, sha, *, merged_by=None, base=None):
        """Record the PR as merged the way GitHub reports a squash merge."""
        self.state = "MERGED"
        self.merged = True
        self.merge_commit_sha = sha
        if merged_by is not None:
            self.merged_by = merged_by
        if base is not None:
            self.base = base
        return self

    def __call__(self, endpoint, *, query=None, paginate=False, profile_home=None):
        self.calls.append(endpoint)
        hook = next((fn for key, fn in self.hooks.items() if key in endpoint), None)
        if endpoint == "graphql":
            value = {"data": {"repository": {"pullRequest": {
                "headRefOid": self.head, "baseRefName": self.base, "state": self.state,
                "baseRef": {"branchProtectionRule": None}}}}}
        elif "/rules/branches/" in endpoint:
            if self.rules_forbidden:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            value = [[]]
        elif "/check-runs" in endpoint:
            runs = list(self.check_runs)
            value = [{"total_count": len(runs), "check_runs": runs}]
        elif "/statuses" in endpoint:
            value = [[]]
        elif "/pulls/" in endpoint:
            if self.pull_read_fails:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            value = {
                "head": {"sha": self.head}, "base": {"ref": self.base},
                "state": "open" if self.state == "OPEN" else "closed",
                "merged": self.merged, "merge_commit_sha": self.merge_commit_sha,
                "merged_by": self.merged_by if self.merged else None,
                "merged_at": "2026-10-01T12:00:00Z" if self.merged else None,
            }
        else:
            raise AssertionError(f"unexpected endpoint {endpoint}")
        if hook is not None:
            hook()
        return value


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(pra, "_api", fake)
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    return fake


def _declare_checks(entry):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"kanban": {"completion_checks": entry}}))


# --------------------------------------------------------------------------
# Real git: a bare origin plus the clone the gate proves the merge against
# --------------------------------------------------------------------------

class Clone:
    """A working clone of a temporary bare origin carrying a ``develop`` branch."""

    def __init__(self, path: Path, origin: Path):
        self.path = path
        self.origin = origin

    def git(self, *args, check=True):
        return subprocess.run(["git", "-C", str(self.path), *args], check=check,
                              capture_output=True, text=True)

    def commit(self, message, *, push_to="develop"):
        """A commit, pushed to the origin unless ``push_to`` is None."""
        self.git("commit", "-q", "--allow-empty", "-m", message)
        sha = self.git("rev-parse", "HEAD").stdout.strip()
        if push_to:
            self.git("push", "-q", "origin", f"HEAD:refs/heads/{push_to}")
        return sha

    def break_origin(self):
        self.git("remote", "set-url", "origin", str(self.origin) + "-deleted")


@pytest.fixture
def clone(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "develop", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True,
                   capture_output=True, text=True)
    repo = Clone(work, origin)
    repo.git("config", "user.email", "gate@example.test")
    repo.git("config", "user.name", "Gate Test")
    repo.commit("base of develop")
    return repo


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------

def _accepted_implementation(conn, github, *, title="Implementation"):
    """A done Implementation card pinned to ``PR_URL`` with an accepted
    exact-head ``pr_acceptance`` receipt — the evidence the gate reads."""
    github.green()
    tid = kb.create_task(conn, title=title, completion_contract="acme/repo")
    assert kb.complete_task(conn, tid, summary="handoff", metadata={"published_pr": PR_URL}) is True
    return tid


def _passing_qa(conn, parent, *, decision="PASS", revision=HEAD, summary="QA done", title="QA"):
    """A done QA card whose latest completed run carries the structured verdict."""
    tid = kb.create_task(conn, title=title)
    kb.link_tasks(conn, parent, tid)
    metadata = {}
    if decision is not None:
        metadata["decision"] = decision
    if revision is not None:
        metadata["revision"] = revision
    assert kb.complete_task(conn, tid, summary=summary, metadata=metadata or None) is True
    return tid


def _gate(conn, clone, impl, qa, *, branch="develop", remote="origin", title="Integrate"):
    gate_id = kb.create_task(conn, title=title)
    kb.link_tasks(conn, impl, gate_id)
    kb.link_tasks(conn, qa, gate_id)
    gates.configure_gate(
        conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
        repository_path=str(clone.path), integration_remote=remote,
        integration_branch=branch,
    )
    return gate_id


def _graph(conn, github, clone, **gate_kwargs):
    """The full opt-in shape: impl -> qa, (impl, qa) -> gate -> downstream."""
    impl = _accepted_implementation(conn, github)
    qa = _passing_qa(conn, impl)
    gate_id = _gate(conn, clone, impl, qa, **gate_kwargs)
    child = kb.create_task(conn, title="Downstream implementation")
    kb.link_tasks(conn, gate_id, child)
    return impl, qa, gate_id, child


def _receipt(conn, task_id):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='integration_acceptance' "
        "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def _condition(receipt, name):
    return next(c for c in receipt["conditions"] if c["name"] == name)


# --------------------------------------------------------------------------
# The happy path and the promotion it unlocks
# --------------------------------------------------------------------------

def test_human_squash_merge_present_in_the_fetched_branch_completes_the_gate(github, clone):
    """Every condition proven: the gate completes and promotes its child."""
    with connect_closing() as conn:
        impl, qa, gate_id, child = _graph(conn, github, clone)
        # The squash merge rewrote the commit: only merge_commit_sha is in develop.
        merge_sha = clone.commit("squash merge of #7")
        github.merge(merge_sha)

        assert kb.get_task(conn, child).status == "todo"
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
        assert kb.get_task(conn, gate_id).status == "done"
        # Gate completion is what promotes the downstream card.
        assert kb.get_task(conn, child).status == "ready"
        receipt = _receipt(conn, gate_id)

    assert receipt["ok"] and receipt["phase"] == "integrated"
    assert receipt["merge_commit_sha"] == merge_sha
    assert receipt["fetched_branch_tip"] == merge_sha
    assert receipt["base_ref"] == "develop"
    assert receipt["merged_by"] == {"login": "maintainer", "type": "User"}
    assert receipt["qa_verdict"] == "PASS"
    assert receipt["qa_revision"] == HEAD == receipt["accepted_head_sha"]
    assert receipt["pr_url"] == PR_URL
    # Secret-free: no gh/git stderr is ever persisted on the receipt.
    assert "stderr" not in json.dumps(receipt)
    assert all(c["ok"] for c in receipt["conditions"])


def test_the_merge_commit_is_proven_not_the_pr_head(github, clone):
    """A squash-merged PR's head is NOT in the branch; requiring it would make
    every squash merge unprovable, so the gate must not look for it."""
    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        merge_sha = clone.commit("squash merge of #7")
        github.merge(merge_sha)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
    # The accepted head (the PR's own tip) is not a commit in this repo at all.
    assert clone.git("cat-file", "-e", HEAD, check=False).returncode != 0


# --------------------------------------------------------------------------
# Fail-closed: one test per condition the mandate names
# --------------------------------------------------------------------------

def test_qa_fail_blocks_the_gate(github, clone):
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl, decision="FAIL")
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "qa_verdict_pass"
    assert not _condition(receipt, "qa_verdict_pass")["ok"]


def test_qa_pass_on_the_wrong_revision_blocks_the_gate(github, clone):
    """QA must have reviewed the exact head GitHub acceptance passed."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl, revision=OTHER_HEAD)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "qa_revision_matches_accepted_head"
    assert receipt["accepted_head_sha"] == HEAD and receipt["qa_revision"] == OTHER_HEAD


def test_prose_only_pass_without_structured_metadata_blocks_the_gate(github, clone):
    """"Everything PASSED!" in a summary is not a verdict."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl, decision=None, revision=None,
                         summary=f"Reviewed {HEAD}: everything PASSED, shipping it")
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "qa_verdict_pass"
    assert receipt["qa_verdict"] is None
    # The receipt says WHY the prose did not count.
    assert "prose" in _condition(receipt, "qa_verdict_pass")["detail"]


def test_an_open_pr_blocks_the_gate_as_waiting_for_the_merge(github, clone):
    with connect_closing() as conn:
        _, _, gate_id, child = _graph(conn, github, clone)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        assert kb.get_task(conn, child).status == "todo"
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "waiting_for_merge"
    assert receipt["phase"] not in UNPROVABLE_PHASES
    assert "Merge the implementation PR" in receipt["recovery"]


def test_a_merge_into_the_wrong_base_blocks_the_gate(github, clone):
    """The PR was merged — into main, not the configured integration branch."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa, branch="develop")
        github.merge(clone.commit("merged into the wrong branch"), base="main")
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "pr_base_is_integration_branch"
    assert receipt["base_ref"] == "main" and receipt["integration_branch"] == "develop"


@pytest.mark.parametrize("actor", [
    {"login": "dependabot[bot]", "type": "User"},
    {"login": "github-actions", "type": "Bot"},
    None,
])
def test_a_bot_merge_blocks_the_gate_under_the_human_policy(github, clone, actor):
    """Both signals GitHub gives for a non-human merger fail closed, and so
    does an unknown actor."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        merge_sha = clone.commit("squash merge of #7")
        github.merge(merge_sha, merged_by=actor)
        if actor is None:
            github.merged_by = None
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "human_merge_actor"


def test_nothing_can_opt_a_gate_out_of_the_human_merge(github, clone):
    """The human merger is MANDATORY: there is no flag, no keyword and no stored
    column that accepts a bot merge, because automation merging its own
    unreviewed work is the failure the gate exists to catch."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"),
                     merged_by={"login": "github-actions[bot]", "type": "Bot"})

        # No caller-side opt-out exists at any layer.
        with pytest.raises(TypeError):
            gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                 repository_path=str(clone.path), require_human_merge=False)

        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "human_merge_actor"
    assert receipt["require_human_merge"] is True
    assert "always requires a human merger" in _condition(receipt, "human_merge_actor")["detail"]


def test_a_row_stored_with_the_retired_opt_out_is_still_gated_on_a_human(github, clone):
    """Defensive against a board an earlier build wrote: the declaration column
    survives for schema compatibility, but a stored ``0`` must not weaken the
    gate — nothing reads it, and re-declaring repairs the row."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        conn.execute("UPDATE integration_gates SET require_human_merge = 0 WHERE gate_task_id = ?",
                     (gate_id,))
        conn.commit()
        assert gates.get_gate(conn, gate_id).require_human_merge is True

        github.merge(clone.commit("squash merge of #7"),
                     merged_by={"login": "release-bot", "type": "Bot"})
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        assert kb.get_task(conn, gate_id).status != "done"
        assert _receipt(conn, gate_id)["phase"] == "human_merge_actor"

        # Re-declaring writes the mandatory value back over the legacy one.
        gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                             repository_path=str(clone.path))
        assert conn.execute(
            "SELECT require_human_merge FROM integration_gates WHERE gate_task_id = ?",
            (gate_id,)).fetchone()[0] == 1


def test_the_cli_offers_no_bot_merge_bypass(github, clone):
    """The retired ``--allow-bot-merge`` must not come back as a public flag."""
    from hermes_cli import kanban as kc

    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = kb.create_task(conn, title="Integrate")
        kb.link_tasks(conn, impl, gate_id)
        kb.link_tasks(conn, qa, gate_id)

    base = (f"integration-gate configure {gate_id} --implementation {impl} --qa {qa} "
            f"--repo {clone.path}")
    refused = kc.run_slash(f"{base} --allow-bot-merge")
    assert "usage error" in refused and "unrecognized arguments" in refused
    assert "--allow-bot-merge" not in kc.run_slash("integration-gate configure --help")

    assert "Integration gate configured" in kc.run_slash(base)
    assert "human merge:    required (always" in kc.run_slash(f"integration-gate show {gate_id}")


@pytest.mark.parametrize("sha", [None, "", "not-a-sha", "a" * 39, "z" * 40])
def test_a_missing_or_malformed_merge_sha_blocks_the_gate(github, clone, sha):
    """Without a usable merge commit the merge cannot be located at all."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        clone.commit("branch moved on without the PR")
        github.merge(sha)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "merge_commit_sha_valid"


def test_a_fetch_failure_blocks_the_gate_as_unprovable(github, clone):
    """A dead remote is "we could not look", never "not merged"."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        clone.break_origin()
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "fetch_failed"
    assert receipt["phase"] in UNPROVABLE_PHASES
    assert receipt["fetched_branch_tip"] is None
    # Nothing of git's stderr (which can carry host/credential detail) is kept.
    assert "does not appear to be a git repository" not in json.dumps(receipt)


def test_a_merge_commit_absent_from_the_fetched_branch_blocks_the_gate(github, clone):
    """GitHub says merged, but that commit is not in origin/develop — the gate
    believes the branch, not the API."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        tip = clone.commit("unrelated work on develop")
        unpushed = clone.commit("merge that never reached origin", push_to=None)
        github.merge(unpushed)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "merge_commit_in_integration_branch"
    # A decided "no" is not the same as an unprovable ancestry question.
    assert receipt["phase"] not in UNPROVABLE_PHASES
    assert receipt["fetched_branch_tip"] == tip


def test_an_undecidable_ancestry_question_is_unprovable_not_a_denial(github, clone):
    """git cannot answer for an object it does not have; that must not read as
    "not merged" (exit 128, not 1)."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        clone.commit("develop moves on")
        github.merge(UNKNOWN_SHA)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "ancestry_unprovable"
    assert receipt["phase"] in UNPROVABLE_PHASES


@pytest.mark.parametrize("break_git", ["missing_binary", "timeout"])
def test_git_being_unusable_is_a_decline_not_a_raise(clone, monkeypatch, break_git):
    """``_git`` is the only thing standing between a missing git binary (or a
    hung fetch) and a traceback out of ``complete_task``. Both must come back
    as a non-zero result the caller can turn into an unproven condition — and
    never as exit 1, which ``is-ancestor`` means as "decided: no"."""
    from hermes_cli import kanban_integration_gate as gate

    if break_git == "missing_binary":
        monkeypatch.setenv("PATH", "")
    else:
        monkeypatch.setattr(gate, "_GIT_TIMEOUT", 0)

    result = gate._git(str(clone.path), "rev-parse", "HEAD")
    assert result.returncode == gate._GIT_UNAVAILABLE != 1
    assert not result.stdout and not result.stderr


def test_an_unusable_git_blocks_the_gate_with_a_receipt(github, clone, monkeypatch):
    """End to end: the decline above becomes a blocking receipt, and the gate
    never claims the merge is absent from a branch it could not read."""
    from hermes_cli import kanban_integration_gate as gate

    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))

        monkeypatch.setattr(gate, "_GIT_TIMEOUT", 0)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        assert kb.get_task(conn, gate_id).status != "done"
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "fetch_failed"
    assert receipt["phase"] in UNPROVABLE_PHASES
    assert not any(c["name"] == "merge_commit_in_integration_branch"
                   for c in receipt["conditions"])


@pytest.mark.parametrize("merged_by", ["octocat", 42, [], {}])
def test_an_unparseable_merger_fails_the_human_condition_closed(github, clone, merged_by):
    """GitHub's merged_by is an object or null; anything else is "no known
    actor", never an implicit human."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        github.merged_by = merged_by
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "human_merge_actor"


def test_a_malformed_acceptance_receipt_blocks_the_gate_instead_of_raising(github, clone):
    """The accepted head is JSON read back off the event log; a non-string
    there must not blow up the regex checks mid-completion."""
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'pr_acceptance'",
            (json.dumps({"ok": True, "head_sha": 12345, "pr_url": None}), impl))
        conn.commit()
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "accepted_head_known"


def test_an_unreadable_pr_is_unprovable_and_never_waiting_for_merge(github, clone):
    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.pull_read_fails = True
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "pr_unreadable"
    assert receipt["phase"] in UNPROVABLE_PHASES
    assert receipt["merge_commit_sha"] is None and receipt["merged_by"] is None


def test_an_implementation_without_an_accepted_receipt_blocks_the_gate(github, clone):
    """Archived (not done) satisfies plain dependency gating, but the gate has
    no accepted exact head to compare QA's revision against."""
    with connect_closing() as conn:
        impl = kb.create_task(conn, title="Never accepted")
        assert kb.archive_task(conn, impl) is True
        qa = _passing_qa(conn, impl)
        gate_id = _gate(conn, clone, impl, qa)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        receipt = _receipt(conn, gate_id)
    assert receipt["phase"] == "implementation_done"


def test_a_failed_gate_leaves_diagnostics_without_making_anything_executable(github, clone):
    """The mandate's core safety property: a refusal is auditable and inert."""
    with connect_closing() as conn:
        impl, qa, gate_id, child = _graph(conn, github, clone)
        before = kb.get_task(conn, gate_id).status
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False

        gate_task = kb.get_task(conn, gate_id)
        assert gate_task.status == before != "done"
        assert kb.get_task(conn, child).status == "todo"
        # The refusal is recorded on the card an operator reads.
        assert "Integration gate waiting_for_merge" in gate_task.last_failure_error
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (gate_id,)).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (gate_id,)))
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (gate_id,)))

    diagnostics = kd.compute_task_diagnostics(row, events, runs)
    blocked = [d for d in diagnostics if d.kind == "integration_gate_blocked"]
    assert len(blocked) == 1
    # Waiting for a human to merge is not an operator emergency.
    assert blocked[0].severity == "warning"
    assert blocked[0].data["phase"] == "waiting_for_merge"
    assert blocked[0].data["unprovable"] is False
    assert blocked[0].data["failing_condition"] == "implementation_pr_merged"
    assert any("integration-gate show" in (a.payload.get("command") or "")
               for a in blocked[0].actions)


def test_diagnostics_separate_an_unprovable_gate_from_one_waiting_for_a_merge(github, clone):
    """"GitHub could not answer" needs an operator; "nobody merged yet" does not."""
    with connect_closing() as conn:
        impl, qa, gate_id, _ = _graph(conn, github, clone)
        github.pull_read_fails = True
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (gate_id,)).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (gate_id,)))

    diagnostic = next(d for d in kd.compute_task_diagnostics(row, events, [])
                      if d.kind == "integration_gate_blocked")
    assert diagnostic.severity == "error"
    assert diagnostic.data["unprovable"] is True
    assert "cannot be proven" in diagnostic.title
    assert "gh authentication" in diagnostic.detail


def test_a_completed_gate_raises_no_blocked_diagnostic(github, clone):
    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (gate_id,)).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (gate_id,)))
    assert not [d for d in kd.compute_task_diagnostics(row, events, [])
                if d.kind == "integration_gate_blocked"]


# --------------------------------------------------------------------------
# Graph semantics: what the gate changes, and what it must not
# --------------------------------------------------------------------------

def test_implementation_done_promotes_qa_but_never_the_gates_child(github, clone):
    """Legacy promotion still flows parent->child one edge at a time; the
    downstream card waits for the GATE, not for the implementation."""
    with connect_closing() as conn:
        impl, qa, gate_id, child = _graph(conn, github, clone)
        # impl and qa are already done via the helpers; the gate became ready.
        assert kb.get_task(conn, impl).status == "done"
        assert kb.get_task(conn, qa).status == "done"
        assert kb.get_task(conn, gate_id).status == "ready"
        assert kb.get_task(conn, child).status == "todo"

        # A second graph proves the QA promotion step itself.
        impl2 = kb.create_task(conn, title="impl2", completion_contract="local-only")
        qa2 = kb.create_task(conn, title="qa2")
        kb.link_tasks(conn, impl2, qa2)
        assert kb.get_task(conn, qa2).status == "todo"
        assert kb.complete_task(conn, impl2, summary="done") is True
        assert kb.get_task(conn, qa2).status == "ready"


def test_legacy_links_keep_their_done_parent_semantics(github, clone):
    """An undeclared card is untouched by the feature: no gate row names it, so
    its child promotes on the parent being done exactly as before."""
    with connect_closing() as conn:
        parent = kb.create_task(conn, title="Plain parent", completion_contract="local-only")
        child = kb.create_task(conn, title="Plain child")
        kb.link_tasks(conn, parent, child)
        assert gates.get_gate(conn, parent) is None
        assert kb.complete_task(conn, parent, summary="no gate here") is True
        assert kb.get_task(conn, child).status == "ready"
        assert _receipt(conn, parent) is None
    assert github.calls == []


def test_an_empty_integration_gates_table_is_inert(github, clone):
    """The opt-in guarantee: with no declarations the schema changes nothing."""
    with connect_closing() as conn:
        assert gates.list_gates(conn) == []
        parent = kb.create_task(conn, title="Parent", completion_contract="local-only")
        child = kb.create_task(conn, title="Child")
        kb.link_tasks(conn, parent, child)
        assert kb.complete_task(conn, parent, summary="ordinary") is True
        assert kb.get_task(conn, child).status == "ready"
        assert conn.execute("SELECT count(*) FROM integration_gates").fetchone()[0] == 0


def test_a_scheduled_child_stays_scheduled_when_the_gate_completes(github, clone):
    """Recompute promotes todo/blocked only — a card parked on a clock must not
    be dragged into the ready queue by its parent finishing."""
    with connect_closing() as conn:
        _, _, gate_id, child = _graph(conn, github, clone)
        assert kb.schedule_task(conn, child, reason="waiting for the release window") is True
        assert kb.get_task(conn, child).status == "scheduled"

        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "scheduled"


def test_configuring_a_gate_mutates_no_card_and_requires_the_real_parents(github, clone):
    """Declaring a gate is a declaration: it never rewrites an existing board's
    graph or moves a card."""
    with connect_closing() as conn:
        impl = kb.create_task(conn, title="impl", completion_contract="local-only")
        qa = kb.create_task(conn, title="qa")
        gate_id = kb.create_task(conn, title="gate")

        # Neither parent is linked yet: the error names the fix and writes nothing.
        with pytest.raises(gates.IntegrationGateConfigError) as exc:
            gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                 repository_path=str(clone.path))
        assert "hermes kanban link" in str(exc.value)
        assert gates.get_gate(conn, gate_id) is None

        kb.link_tasks(conn, impl, gate_id)
        with pytest.raises(gates.IntegrationGateConfigError):
            gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                 repository_path=str(clone.path))

        kb.link_tasks(conn, qa, gate_id)
        # Snapshot the graph the declaration must leave exactly as it found it.
        before = {t: kb.get_task(conn, t) for t in (impl, qa, gate_id)}
        links_before = set(conn.execute("SELECT parent_id, child_id FROM task_links"))

        config = gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                      repository_path=str(clone.path))
        assert config.integration_branch == "develop"
        assert config.require_human_merge is True
        # No card changed status or assignee, and no edge was created.
        for task_id, snapshot in before.items():
            now = kb.get_task(conn, task_id)
            assert (now.status, now.assignee) == (snapshot.status, snapshot.assignee)
        assert set(conn.execute("SELECT parent_id, child_id FROM task_links")) == links_before
        assert [c.gate_task_id for c in gates.list_gates(conn)] == [gate_id]

        # Re-declaring is an update, not a duplicate row.
        updated = gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                       repository_path=str(clone.path), integration_branch="main")
        assert updated.integration_branch == "main" and updated.require_human_merge is True
        assert len(gates.list_gates(conn)) == 1

        # Removing it returns the card to ordinary parent gating.
        assert gates.remove_gate(conn, gate_id) is True
        assert gates.get_gate(conn, gate_id) is None
        assert gates.remove_gate(conn, gate_id) is False


@pytest.mark.parametrize("remote,branch", [
    ("--upload-pack=evil", "develop"), ("origin", "../../etc"), ("origin", "a branch"),
    ("origin", "develop^"), ("", "develop"),
])
def test_a_remote_or_branch_that_is_not_a_plain_ref_name_is_refused(github, clone, remote, branch):
    """Both are interpolated into a git refspec, so neither may carry an option
    or a traversal."""
    with connect_closing() as conn:
        impl = kb.create_task(conn, title="impl", completion_contract="local-only")
        qa = kb.create_task(conn, title="qa")
        gate_id = kb.create_task(conn, title="gate")
        kb.link_tasks(conn, impl, gate_id)
        kb.link_tasks(conn, qa, gate_id)
        with pytest.raises(gates.IntegrationGateConfigError):
            gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                 repository_path=str(clone.path), integration_remote=remote,
                                 integration_branch=branch)
        assert gates.get_gate(conn, gate_id) is None


def test_a_relative_or_missing_repository_path_is_refused(github, clone, tmp_path):
    with connect_closing() as conn:
        impl = kb.create_task(conn, title="impl", completion_contract="local-only")
        qa = kb.create_task(conn, title="qa")
        gate_id = kb.create_task(conn, title="gate")
        kb.link_tasks(conn, impl, gate_id)
        kb.link_tasks(conn, qa, gate_id)
        for bad in ("relative/path", str(tmp_path / "nope")):
            with pytest.raises(gates.IntegrationGateConfigError):
                gates.configure_gate(conn, gate_id, implementation_task_id=impl, qa_task_id=qa,
                                     repository_path=bad)
        assert gates.get_gate(conn, gate_id) is None


def test_a_declaration_survives_archive_and_is_purged_by_a_hard_delete(github, clone):
    """Archiving is recoverable, so the declaration must survive it; the hard
    delete that drops a card's links drops its gate row too, leaving no row
    pointing at a task that no longer exists."""
    with connect_closing() as conn:
        impl, qa, gate_id, _ = _graph(conn, github, clone)
        assert kb.archive_task(conn, gate_id) is True
        assert gates.get_gate(conn, gate_id) is not None

        assert kb.delete_archived_task(conn, gate_id) is True
        assert gates.get_gate(conn, gate_id) is None
        assert gates.list_gates(conn) == []


def test_deleting_a_parent_of_a_gate_purges_the_declaration_too(github, clone):
    """A gate whose implementation or QA card is gone can never be proven, so
    no orphaned declaration is left behind to block the card forever."""
    with connect_closing() as conn:
        impl, qa, gate_id, _ = _graph(conn, github, clone)
        assert kb.archive_task(conn, qa) is True
        assert kb.delete_archived_task(conn, qa) is True
        assert gates.get_gate(conn, gate_id) is None


# --------------------------------------------------------------------------
# Snapshot discipline around the external evidence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("interference", ["claim", "reconfigure", "remove_gate"])
def test_a_board_change_during_verification_rejects_the_snapshot(github, clone, interference):
    """Network and git work happen with no write txn open, so the board can
    move underneath them. The recheck inside ``complete_task``'s transaction
    must refuse rather than promote on evidence computed from a stale board.
    """
    with connect_closing() as conn:
        impl, qa, gate_id, child = _graph(conn, github, clone)
        merge_sha = clone.commit("squash merge of #7")
        github.merge(merge_sha)

        def interfere():
            with connect_closing() as rival:
                if interference == "claim":
                    assert kb.claim_task(rival, gate_id) is not None
                elif interference == "reconfigure":
                    gates.configure_gate(
                        rival, gate_id, implementation_task_id=impl, qa_task_id=qa,
                        repository_path=str(clone.path), integration_branch="main")
                else:
                    assert gates.remove_gate(rival, gate_id) is True

        # Fires while the gate is reading GitHub, i.e. after prepare_* snapshotted.
        github.hooks["/pulls/"] = interfere
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False

        assert kb.get_task(conn, gate_id).status != "done"
        assert kb.get_task(conn, child).status == "todo"
        # A rejected snapshot writes no receipt at all — the evidence it would
        # describe was never valid for the board as it now stands.
        assert _receipt(conn, gate_id) is None


@pytest.mark.parametrize("mutation", ["qa_verdict", "qa_revision", "contract", "acceptance"])
def test_mutating_the_evidence_during_verification_refuses_the_completion(github, clone, mutation):
    """The evidence the gate approves from is MUTABLE while it reads GitHub.

    Comparing only run ids and statuses was not enough:
    ``edit_task(result=…)`` is the supported way to rewrite a completed
    card's result, and it changes the completed QA run's summary and metadata
    while leaving every id and status exactly as the snapshot recorded them. The
    implementation's pinned contract and its accepted ``pr_acceptance`` receipt
    are ordinary rows too. Each of these would otherwise let a gate complete —
    and promote its downstream card — on a verdict, a reviewed revision or an
    accepted head that no longer exists.
    """
    with connect_closing() as conn:
        impl, qa, gate_id, child = _graph(conn, github, clone)
        github.merge(clone.commit("squash merge of #7"))

        def interfere():
            with connect_closing() as rival:
                if mutation == "qa_verdict":
                    # Same run, same statuses — only the verdict flips.
                    assert kb.edit_task(
                        rival, qa, result="actually failing",
                        metadata={"decision": "FAIL", "revision": HEAD}) is True
                elif mutation == "qa_revision":
                    assert kb.edit_task(
                        rival, qa, result="reviewed something else",
                        metadata={"decision": "PASS", "revision": OTHER_HEAD}) is True
                elif mutation == "contract":
                    rival.execute("UPDATE tasks SET completion_contract = ? WHERE id = ?",
                                  ("https://github.com/acme/repo/pull/8", impl))
                    rival.commit()
                else:
                    rival.execute(
                        "UPDATE task_events SET payload = ? WHERE task_id = ? "
                        "AND kind = 'pr_acceptance'",
                        (json.dumps({"ok": True, "head_sha": OTHER_HEAD, "pr_url": PR_URL}), impl))
                    rival.commit()

        # Fires while the gate is reading GitHub, i.e. after prepare_* snapshotted.
        github.hooks["/pulls/"] = interfere
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False

        assert kb.get_task(conn, gate_id).status != "done"
        assert kb.get_task(conn, child).status == "todo"
        # No receipt at all, and in particular no accepted one: the evidence the
        # verification would describe was never valid for the board as it stands.
        receipts = [json.loads(r["payload"]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='integration_acceptance'",
            (gate_id,))]
        assert receipts == []


def test_the_gate_holds_no_write_lock_while_it_reads_github_and_git(github, clone):
    """A concurrent writer must be able to commit mid-verification; if the gate
    held the board's write lock this would raise "database is locked"."""
    observed = []

    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        merge_sha = clone.commit("squash merge of #7")
        github.merge(merge_sha)

        def write_from_another_connection():
            with connect_closing() as rival:
                other = kb.create_task(rival, title="written mid-verification")
                observed.append(kb.get_task(rival, other).id)

        github.hooks["/pulls/"] = write_from_another_connection
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True

    assert len(observed) == 1


def test_a_stale_expected_run_id_refuses_before_any_external_call(github, clone):
    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        github.merge(clone.commit("squash merge of #7"))
        calls_before = len(github.calls)
        assert kb.complete_task(conn, gate_id, summary="gate completion", expected_run_id=987654) is False
        assert _receipt(conn, gate_id) is None
    assert github.calls[calls_before:] == []


def test_a_gate_receipt_is_immutable_history_across_attempts(github, clone):
    """Each attempt appends; a later success never rewrites the refusals."""
    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
        receipts = [json.loads(r["payload"]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='integration_acceptance' "
            "ORDER BY id", (gate_id,))]
    assert [r["ok"] for r in receipts] == [False, True]
    assert [r["phase"] for r in receipts] == ["waiting_for_merge", "integrated"]


# --------------------------------------------------------------------------
# Operator-facing rendering
# --------------------------------------------------------------------------

def test_describe_receipt_shows_every_condition_and_the_stopping_point(github, clone):
    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
        blocked = _receipt(conn, gate_id)
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True
        passed = _receipt(conn, gate_id)

    blocked_text = "\n".join(describe_receipt(blocked))
    assert "BLOCKED at waiting_for_merge" in blocked_text
    assert "✗ implementation_pr_merged" in blocked_text
    assert "✓ qa_verdict_pass" in blocked_text
    assert "Next step:" in blocked_text

    passed_text = "\n".join(describe_receipt(passed))
    assert "every condition proven" in passed_text
    assert "✗" not in passed_text
    assert "merge commit:" in passed_text

    assert "none yet" in "\n".join(describe_receipt(None))


def test_the_natural_complete_failure_names_the_gate_that_refused(github, clone):
    """``hermes kanban complete`` on a refused gate used to say "unknown id or
    terminal state", which is both wrong and the opposite of actionable — the
    card exists and is perfectly completable, a condition just is not proven.
    The real phase and recovery are already on the receipt the refusal wrote."""
    from hermes_cli import kanban as kc

    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)

    waiting = kc.run_slash(f"complete {gate_id} --summary 'gate completion'")
    assert "unknown id or terminal state" not in waiting
    assert "integration gate refused — waiting_for_merge." in waiting
    assert "is open and not merged." in waiting
    assert "Merge the implementation PR" in waiting

    # An unprovable refusal reports its own phase, not the merge-wait one.
    github.pull_read_fails = True
    unprovable = kc.run_slash(f"complete {gate_id} --summary 'gate completion'")
    assert "integration gate refused — pr_unreadable." in unprovable
    assert "gh authentication" in unprovable

    # The generic message is still what a genuinely unknown id gets.
    assert "unknown id or terminal state" in kc.run_slash("complete t_deadbeef --summary 'nothing here'")


def test_a_pr_acceptance_refusal_also_reports_its_real_phase(github, clone):
    """The same for the other completion gate: a red required check is reported
    as a red required check."""
    from hermes_cli import kanban as kc

    github.check_runs = [{
        "id": 43, "name": "ci/test", "head_sha": HEAD, "app": {"id": 1},
        "status": "completed", "conclusion": "failure",
        "html_url": "https://github.com/acme/repo/actions/runs/43",
    }]
    with connect_closing() as conn:
        tid = kb.create_task(conn, title="Implementation", completion_contract="acme/repo")

    refused = kc.run_slash(f"complete {tid} --summary 'implemented' --metadata '{{\"published_pr\": \"{PR_URL}\"}}'")
    assert "unknown id or terminal state" not in refused
    # The verdict, not just the phase collection happened to stop at.
    assert "PR acceptance refused — failure at the " in refused
    assert "Fix required failures" in refused


def test_a_refusal_with_no_receipt_keeps_the_generic_message(github, clone):
    """The receipt floor matters: a card refused for an ordinary reason must not
    be explained with a gate receipt an EARLIER attempt left behind."""
    from hermes_cli import kanban as kc

    with connect_closing() as conn:
        _, _, gate_id, _ = _graph(conn, github, clone)
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False  # leaves a receipt
        assert _receipt(conn, gate_id) is not None
        # Now refuse for a reason no gate explains: the gate is already done.
        github.merge(clone.commit("squash merge of #7"))
        assert kb.complete_task(conn, gate_id, summary="gate completion") is True

    already_done = kc.run_slash(f"complete {gate_id} --summary 'gate completion'")
    assert "unknown id or terminal state" in already_done
    assert "refused" not in already_done


def test_the_cli_declares_inspects_and_removes_a_gate(github, clone):
    """The configuration surface end to end through the real argparse tree."""
    from hermes_cli import kanban as kc

    with connect_closing() as conn:
        impl = _accepted_implementation(conn, github)
        qa = _passing_qa(conn, impl)
        gate_id = kb.create_task(conn, title="Integrate")
        kb.link_tasks(conn, impl, gate_id)
        kb.link_tasks(conn, qa, gate_id)

    assert "No integration gates are declared" in kc.run_slash("integration-gate list")

    configured = kc.run_slash(
        f"integration-gate configure {gate_id} --implementation {impl} --qa {qa} "
        f"--repo {clone.path} --branch develop")
    assert f"Integration gate configured on {gate_id}" in configured
    assert "origin/develop" in configured

    listed = kc.run_slash("integration-gate ls")
    assert gate_id in listed and "never verified" in listed

    shown = json.loads(kc.run_slash(f"integration-gate show {gate_id} --json"))
    assert shown["config"]["integration_branch"] == "develop"
    assert shown["config"]["require_human_merge"] is True
    assert shown["latest_verification"] is None

    # A refusal becomes visible on the same surface.
    with connect_closing() as conn:
        assert kb.complete_task(conn, gate_id, summary="gate completion") is False
    assert "BLOCKED at waiting_for_merge" in kc.run_slash(f"integration-gate show {gate_id}")
    assert "blocked at waiting_for_merge" in kc.run_slash("integration-gate list")

    assert "ordinary card again" in kc.run_slash(f"integration-gate rm {gate_id}")
    with connect_closing() as conn:
        assert gates.get_gate(conn, gate_id) is None


def test_the_cli_reports_a_bad_declaration_without_a_traceback(github, clone):
    from hermes_cli import kanban as kc

    with connect_closing() as conn:
        impl = kb.create_task(conn, title="impl", completion_contract="local-only")
        qa = kb.create_task(conn, title="qa")
        gate_id = kb.create_task(conn, title="gate")

    # Parents are not linked yet: a plain operator-facing error.
    unlinked = kc.run_slash(
        f"integration-gate configure {gate_id} --implementation {impl} --qa {qa} "
        f"--repo {clone.path}")
    assert "hermes kanban link" in unlinked and "Traceback" not in unlinked

    missing = kc.run_slash(f"integration-gate show {gate_id}")
    assert "ordinary card" in missing and "Traceback" not in missing
