"""Acceptance invariants for GitHub completion contracts.

Regression for the two reproduced defects: Repository Rules was read BEFORE
Check Runs and its 403 on a private/free repository aborted collection (the
receipt then claimed ``checks: []`` as if CI had reported nothing), and the
exact PR had to be re-supplied on the reviewer's completion because nothing
bound it at the implementer's review handoff.

GitHub is mocked at ``kanban_pr_acceptance._api`` — the single process
boundary the gate talks to GitHub through — so the real collector, the real
config loader, real SQLite and the real ``complete_task`` /
``request_review`` lifecycle run on every host.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as pra
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_pr_acceptance_store import PublishedPrBindingError

PR_URL = "https://github.com/acme/repo/pull/7"
HEAD = "a" * 40
OTHER_HEAD = "b" * 40


class FakeGitHub:
    """``gh api`` responses in the shapes GitHub actually returns.

    ``rules_forbidden`` reproduces a private repository without a paid plan:
    ``gh`` exits non-zero on the 403 from the Rules endpoint.
    """

    def __init__(self):
        self.head = HEAD
        self.base = "main"
        self.state = "OPEN"
        self.merged = False
        self.protection_contexts = []
        self.rules_contexts = []
        self.rules_forbidden = False
        self.check_runs = []
        self.statuses = []
        self.noise_runs = 0
        self.total_count_override = None
        self.calls = []
        self.hooks = {}

    # --- helpers used by the tests ---

    def run(self, name, conclusion, *, status="completed", head=None, app_id=None, run_id=42):
        self.check_runs.append({
            "id": run_id, "name": name, "head_sha": head or self.head,
            "app": {"id": app_id if app_id is not None else 1},
            "status": status, "conclusion": conclusion,
            "html_url": f"https://github.com/acme/repo/actions/runs/{run_id}",
        })
        return self

    def legacy_status(self, context, state, *, status_id=5):
        self.statuses.append({
            "id": status_id, "context": context, "state": state,
            "target_url": "https://ci.example/1",
        })
        return self

    # --- transport ---

    def __call__(self, endpoint, *, query=None, paginate=False):
        self.calls.append(endpoint)
        hook = next((fn for key, fn in self.hooks.items() if key in endpoint), None)
        if endpoint == "graphql":
            protection = (
                {"requiredStatusChecks": [{"context": c, "app": {"databaseId": a}}
                                          for c, a in self.protection_contexts]}
                if self.protection_contexts else None
            )
            value = {"data": {"repository": {"pullRequest": {
                "headRefOid": self.head, "baseRefName": self.base, "state": self.state,
                "baseRef": {"branchProtectionRule": protection}}}}}
        elif "/rules/branches/" in endpoint:
            if self.rules_forbidden:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            value = [[{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": c, "integration_id": a}
                                           for c, a in self.rules_contexts]}}]] \
                if self.rules_contexts else [[]]
        elif "/check-runs" in endpoint:
            value = self._check_run_pages()
        elif "/statuses" in endpoint:
            value = [self.statuses] if self.statuses else [[]]
        elif "/pulls/" in endpoint:
            value = {"head": {"sha": self.head}, "base": {"ref": self.base},
                     "state": "closed" if self.state != "OPEN" else "open",
                     "merged": self.merged}
        else:
            raise AssertionError(f"unexpected endpoint {endpoint}")
        if hook is not None:
            hook()
        return value

    def _check_run_pages(self):
        """Noise runs fill whole 100-item pages so a required run can only be
        found by following pagination past the first page."""
        noise = [{"id": 1000 + i, "name": "noise", "head_sha": self.head,
                  "app": {"id": 1}, "status": "completed", "conclusion": "skipped"}
                 for i in range(self.noise_runs)]
        runs = noise + list(self.check_runs)
        total = self.total_count_override if self.total_count_override is not None else len(runs)
        pages = [runs[i:i + 100] for i in range(0, len(runs), 100)] or [[]]
        return [{"total_count": total, "check_runs": page} for page in pages]


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(pra, "_api", fake)
    return fake


def _declare_checks(entry):
    """Write ``kanban.completion_checks`` to the temp home's real config.yaml."""
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"kanban": {"completion_checks": entry}}))


def _receipt(conn, task_id):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' "
        "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def _card(conn, *, contract="acme/repo", title="Publish"):
    return kb.create_task(conn, title=title, completion_contract=contract)


def test_rules_403_accepts_declared_green_check_and_separates_its_evidence(github):
    """A private/free repository's Rules 403 must not stop the Check Runs read."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is True
        assert kb.get_task(conn, tid).status == "done"
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["phase"] == "accepted"
    assert receipt["head_sha"] == HEAD
    # Rules unreadable is recorded as such, and the checks endpoint was still read.
    assert receipt["rules"]["available"] is False and receipt["rules"]["reason"]
    assert receipt["checks_endpoint"]["fetched"] is True
    assert receipt["required_sources"] == ["configured"]
    assert [c["classification"] for c in receipt["checks"]] == ["success"]


def test_rules_403_without_any_declared_policy_fails_before_the_checks_read(github):
    """No declared policy is fail-closed, and the receipt says so without
    pretending the checks endpoint answered empty."""
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert not receipt["ok"] and receipt["phase"] == "declared_policy"
    assert receipt["rules"]["available"] is False
    assert receipt["checks_endpoint"]["fetched"] is False
    assert receipt["checks"] == []
    assert "completion_checks" in receipt["detail"]
    assert not any("check-runs" in call for call in github.calls)


def test_rules_403_with_declared_policy_and_no_ci_at_all_fails(github):
    """Declared policy + an empty checks endpoint is distinguishable from the
    case above: the endpoint was read and returned nothing."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "missing"
    assert receipt["checks_endpoint"] == {"fetched": True, "total_count": 0, "runs": 0,
                                          "statuses": 0, "pages": 1}
    assert [c["classification"] for c in receipt["checks"]] == ["missing"]


def test_declared_check_absent_while_other_checks_are_green_fails(github):
    """All-green-observed is never a substitute for the declared check."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("some-other-job", "success", run_id=9)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "missing"
    assert [c["name"] for c in receipt["checks"]] == ["ci/test"]


@pytest.mark.parametrize("conclusion,status", [
    ("failure", "completed"), ("cancelled", "completed"), ("timed_out", "completed"),
    ("skipped", "completed"), ("neutral", "completed"), ("action_required", "completed"),
    ("stale", "completed"), (None, "in_progress"), ("some_future_conclusion", "completed"),
])
def test_no_non_success_conclusion_family_can_complete(github, conclusion, status):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", conclusion, status=status)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["classification"] != "success"
    assert receipt["checks"][0]["conclusion"] == conclusion


def test_required_check_past_the_first_page_is_found_and_truncation_fails(github):
    """The required run only exists on page 2 of the paginated response."""
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.noise_runs = 100
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
        assert receipt["checks_endpoint"]["pages"] == 2
        assert receipt["checks_endpoint"]["runs"] == 101

        # A page set that does not add up to total_count is incomplete evidence.
        github.total_count_override = 500
        stale_pages = _card(conn, title="truncated")
        assert kb.complete_task(conn, stale_pages, metadata={"published_pr": PR_URL}) is False
        truncated = _receipt(conn, stale_pages)
    assert truncated["classification"] == "infra" and truncated["phase"] == "check_runs"


def test_legacy_statuses_are_paginated_for_the_exact_head(github):
    _declare_checks({"acme/repo": {"required_checks": ["legacy-ci"]}})
    github.rules_forbidden = True
    github.legacy_status("legacy-ci", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, tid)
    assert receipt["checks_endpoint"]["statuses"] == 1
    assert receipt["checks"][0]["head_sha"] == HEAD


@pytest.mark.parametrize("field,value", [("head", OTHER_HEAD), ("base", "release")])
def test_head_or_base_change_during_collection_fails_stale(github, field, value):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")

    def mutate():
        setattr(github, "head" if field == "head" else "base", value)

    github.hooks["/check-runs"] = mutate
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).status != "done"
        receipt = _receipt(conn, tid)
    assert receipt["classification"] == "stale" and receipt["phase"] == "stale_recheck"


def test_readable_rules_and_branch_protection_contexts_are_still_enforced(github):
    """Readable GitHub policy keeps working, with no config entry at all."""
    github.protection_contexts = [("protected-ci", 1)]
    github.rules_contexts = [("ruleset-ci", None)]
    github.run("protected-ci", "success", run_id=11)
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        missing_ruleset = _receipt(conn, tid)
        assert missing_ruleset["classification"] == "missing"
        assert sorted(missing_ruleset["required_sources"]) == ["branch_protection", "repository_rules"]

        github.run("ruleset-ci", "success", run_id=12)
        done = _card(conn, title="both green")
        assert kb.complete_task(conn, done, metadata={"published_pr": PR_URL}) is True
        receipt = _receipt(conn, done)
    assert receipt["rules"]["available"] is True and receipt["rules"]["reason"] is None
    assert {c["name"] for c in receipt["checks"]} == {"protected-ci", "ruleset-ci"}


def test_review_handoff_pins_the_pr_so_the_reviewer_completes_without_repeating_it(github):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.request_review(conn, tid, summary="implemented", metadata={"published_pr": PR_URL})
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        # The reviewer's completion carries no PR evidence of its own.
        assert kb.complete_task(conn, tid, summary="approved") is True
        receipt = _receipt(conn, tid)
    assert receipt["ok"] and receipt["pr_url"] == PR_URL


def test_pr_named_under_the_wrong_key_is_an_explicit_error_on_both_transitions(github):
    with connect_closing() as conn:
        completing = _card(conn)
        with pytest.raises(PublishedPrBindingError) as complete_err:
            kb.complete_task(conn, completing, metadata={"pr_url": PR_URL})
        assert "published_pr" in str(complete_err.value) and "pr_url" in str(complete_err.value)
        # Nothing was mutated and no GitHub call was made for an unbindable handoff.
        assert kb.get_task(conn, completing).status == "ready"
        assert kb.get_task(conn, completing).completion_contract == "acme/repo"

        reviewing = _card(conn, title="review handoff")
        with pytest.raises(PublishedPrBindingError):
            kb.request_review(conn, reviewing, summary="done", metadata={"pr_url": PR_URL})
        assert kb.get_task(conn, reviewing).status == "ready"
        assert kb.get_task(conn, reviewing).completion_contract == "acme/repo"
    assert github.calls == []


def test_a_sibling_pr_cannot_replace_the_pinned_one_after_a_failed_attempt(github):
    _declare_checks({"acme/repo": {"required_checks": ["ci/test"]}})
    github.rules_forbidden = True
    github.run("ci/test", "failure")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        assert kb.get_task(conn, tid).completion_contract == PR_URL

        # The retry names a different PR of the same repository.
        sibling = "https://github.com/acme/repo/pull/8"
        assert kb.complete_task(conn, tid, metadata={"published_pr": sibling}) is False
        assert kb.get_task(conn, tid).completion_contract == PR_URL
        rejection = _receipt(conn, tid)
        assert rejection["pr_url"] == PR_URL and rejection["rejected_pr"] == sibling

        # And a PR from another repository is rejected outright.
        with pytest.raises(PublishedPrBindingError):
            kb.request_review(conn, tid, summary="x",
                              metadata={"published_pr": "https://github.com/other/repo/pull/7"})

        github.check_runs = []
        github.run("ci/test", "success")
        assert kb.complete_task(conn, tid, summary="green now") is True
        assert kb.get_task(conn, tid).completion_contract == PR_URL


def test_declared_checks_load_from_config_yaml_and_bad_entries_report_themselves(github):
    """The loader's own data flow: real config.yaml -> real load_config()."""
    from hermes_cli.kanban_completion_policy import configured_required_checks

    _declare_checks({"ACME/Repo": ["from-bare-list", "from-bare-list", " spaced "]})
    assert configured_required_checks("acme/repo") == (
        ["from-bare-list", "spaced"], None)
    assert configured_required_checks("other/repo") == ([], None)

    _declare_checks({"acme/repo": {"required_checks": "ci/test"}})
    names, problem = configured_required_checks("acme/repo")
    assert names == [] and "must be a list" in problem

    _declare_checks({"acme/repo": {"note": "todo"}})
    names, problem = configured_required_checks("acme/repo")
    assert names == [] and "required_checks" in problem

    # A malformed entry blocks completion and names itself in the receipt.
    github.rules_forbidden = True
    github.run("ci/test", "success")
    with connect_closing() as conn:
        tid = _card(conn)
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR_URL}) is False
        receipt = _receipt(conn, tid)
    assert receipt["phase"] == "declared_policy"
    assert "required_checks" in receipt["config_problem"]


def test_local_only_cards_never_reach_github(github):
    with connect_closing() as conn:
        tid = _card(conn, contract="local-only")
        assert kb.complete_task(conn, tid, summary=f"{PR_URL} is background context") is True
        assert _receipt(conn, tid) is None
    assert github.calls == []
