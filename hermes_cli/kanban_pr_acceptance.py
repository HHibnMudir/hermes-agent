"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

Evidence is collected in named phases and the receipt reports the phase that
stopped it, because "CI reported nothing" and "we never asked CI" are different
facts. Repository Rules is OPTIONAL evidence: a private repository without
GitHub Pro answers 403 there, and reading it before the Check Runs endpoint
aborted collection on that 403 — the receipt then said ``checks: []`` as if CI
had reported nothing. Rules unavailability is now recorded and collection
continues against the declared policy (``kanban.completion_checks``); every
other endpoint this gate needs stays mandatory and fails closed.
"""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote

from hermes_cli.kanban_completion_policy import CONFIG_DOTPATH, configured_required_checks

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{40}")

# Evidence failures that mean "GitHub could not answer", not "the card is red".
_API_FAILURES = (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError)

_RECOVERY = (
    "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
    "Use kanban_block if human input is needed; receipts remain on the task event log."
)


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def _repository_rules_required(repo: str, branch: str) -> tuple[set, bool, str | None]:
    """``(contexts, available, reason)`` from Repository Rules; never raises.

    Rules is a documented fallback boundary, not an implicit pass: an
    unavailable read only means its contexts cannot be added to the required
    set, and acceptance then rests on whatever policy IS readable (classic
    branch protection) or declared in config.
    """
    try:
        pages = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
        contexts = set()
        for page in pages:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    contexts.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        return contexts, True, None
    except _API_FAILURES:
        # Never persist gh stderr (credentials/host details); "unreadable" is the actionable fact.
        return set(), False, (
            "repository rules unreadable (403 on private repositories without a paid plan, "
            "or no rules read access); required checks fell back to the declared policy"
        )


def collect_acceptance(contract: str, published_pr: str | None) -> dict:
    receipt: dict = {
        "ok": False, "classification": "missing", "phase": "pr_binding",
        "head_sha": None, "base_ref": None, "pr_url": published_pr, "checks": [],
        "required": [], "required_sources": [], "configured_required_checks": [],
        # ``available: None`` = the Rules endpoint was never reached, which is
        # not the same as "Rules said there are no required checks".
        "rules": {"available": None, "reason": None, "contexts": 0},
        # ``fetched: False`` keeps an empty ``checks`` list from reading as
        # "the checks endpoint returned nothing" when collection aborted first.
        "checks_endpoint": {"fetched": False, "total_count": None, "runs": 0, "statuses": 0, "pages": 0},
        "recovery": _RECOVERY,
    }
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            # The receipt names the card's own PR, never the rejected candidate:
            # a pinned card's evidence is its pinned PR even when the handoff
            # offered a sibling.
            receipt["pr_url"] = contract if declared else None
            if published_pr and published_pr != contract:
                receipt["rejected_pr"] = published_pr
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")

        receipt["phase"] = "pr_resolve"
        query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        pr = _api("graphql", query=query)["data"]["repository"]["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"], receipt["base_ref"] = sha, branch
        if not _SHA.fullmatch(sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")

        # Required set = every readable/declared policy, keyed by (context, app id).
        required: dict[tuple, str] = {}
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        for rule in protection.get("requiredStatusChecks", []):
            required.setdefault((rule["context"], (rule.get("app") or {}).get("databaseId")),
                                "branch_protection")
        receipt["phase"] = "repository_rules"
        rules_contexts, rules_available, rules_reason = _repository_rules_required(repo, branch)
        receipt["rules"] = {"available": rules_available, "reason": rules_reason,
                            "contexts": len(rules_contexts)}
        for key in rules_contexts:
            required.setdefault(key, "repository_rules")
        receipt["phase"] = "declared_policy"
        configured, config_problem = configured_required_checks(repo)
        receipt["configured_required_checks"] = configured
        if config_problem:
            receipt["config_problem"] = config_problem
        for context in configured:
            required.setdefault((context, None), "configured")
        receipt["required"] = [{"context": c, "app_id": a, "source": source}
                               for (c, a), source in sorted(required.items(), key=lambda kv: str(kv[0]))]
        receipt["required_sources"] = sorted(set(required.values()))
        if not required:
            receipt["detail"] = _no_policy_detail(repo, rules_available, config_problem)
            return receipt

        receipt["phase"] = "check_runs"
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        receipt["phase"] = "statuses"
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        receipt["checks_endpoint"] = {"fetched": True, "total_count": pages[0]["total_count"],
                                     "runs": len(runs), "statuses": len(statuses), "pages": len(pages)}

        receipt["phase"] = "evaluate"
        outcomes = _evaluate(receipt, required, runs, statuses, sha)

        # Re-read after all pages: old-head successes are never transferable.
        receipt["phase"] = "stale_recheck"
        current = _api(f"repos/{repo}/pulls/{number}")
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        if receipt["ok"]:
            receipt["phase"] = "accepted"
        return receipt
    except _API_FAILURES:
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail=(
            f"GitHub acceptance evidence unavailable or incomplete at the "
            f"{receipt['phase']} phase; check gh authentication/API access and retry."))
        return receipt


def _no_policy_detail(repo: str, rules_available: bool, config_problem: str | None) -> str:
    """Why nothing could be required — the one fail-closed case an operator fixes in config."""
    policy = ("GitHub reports no required checks for this base branch"
              if rules_available else
              "GitHub's repository rules are unreadable for this repository")
    fix = f"declare them under {CONFIG_DOTPATH}: {{{repo!r}: {{required_checks: [...]}}}}"
    if config_problem:
        return f"No required checks are declared: {policy} and {config_problem}. Fix the config entry."
    return (f"No required checks are declared for {repo}: {policy} and no required_checks are "
            f"configured. Either {fix}, or use a local-only contract for non-CI tasks.")


def _evaluate(receipt: dict, required: dict, runs: list, statuses: list, sha: str) -> list[str]:
    """Classify every required context against exact-head evidence."""
    outcomes: list[str] = []
    for (context, app_id), source in sorted(required.items(), key=lambda kv: str(kv[0])):
        matching = [r for r in runs if r["name"] == context and
                    (app_id in (None, -1) or r["app"]["id"] == app_id)]
        # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
        legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
        selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
        if not selected:
            outcomes.append("missing")
            receipt["checks"].append({"name": context, "classification": "missing",
                                      "head_sha": sha, "source": source})
        for check in selected:
            is_run = "conclusion" in check
            outcome = check.get("conclusion") if is_run else check["state"]
            classification = _classify(check, sha, outcome, is_run)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": check["id"],
                "url": check.get("html_url") or check.get("target_url"),
                "head_sha": check.get("head_sha", check.get("sha")),
                "classification": classification, "conclusion": outcome, "source": source})
    return outcomes


# Every GitHub conclusion that is not ``success`` blocks completion; the value
# is kept verbatim in the receipt so a reader can tell a red test from a
# cancelled run from an unknown future conclusion.
_CONCLUSIONS = {
    "success": "success", "failure": "failure", "pending": "pending", "error": "infra",
    "cancelled": "cancelled", "timed_out": "timed_out", "action_required": "action_required",
    "neutral": "neutral", "skipped": "skipped", "stale": "stale", "startup_failure": "failure",
}


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return _CONCLUSIONS.get(outcome, "unknown")
