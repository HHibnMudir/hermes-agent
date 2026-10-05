"""Shape of the GitHub JSON both completion gates consume.

``gh api`` output is JSON decoded from a subprocess, and the gates dereference
it field by field: ``pr["head"]["sha"]``, ``pr["base"]["ref"]``,
``pr["merged"]``. A proxy's error envelope, an enterprise host's older schema or
a future field rename can put a string, a list or ``null`` where an object
belongs — and ``merged: "false"`` is a TRUTHY string, so an untyped read of it
accepts a merge that never happened.

Every field is therefore checked for shape HERE, once, before anything
dereferences it, so a malformed answer blocks the gate with a receipt instead
of raising a ``KeyError``/``TypeError`` out of the middle of a completion. The
problem comes back as a short operator-facing sentence that never quotes the
response body (which can carry tokens or host detail).

Fields only a MERGED pull request carries (``merged_by``,
``merge_commit_sha``, ``merged_at``) are ``null`` on an open PR, so they are
NOT validated here: each is the subject of its own gate condition, which is
what lets a receipt say "merged by a bot" or "no usable merge commit" instead
of a flat "malformed".
"""
from __future__ import annotations

import re

#: A git object name as GitHub reports it: exactly 40 lowercase hex digits.
SHA_RE = re.compile(r"[0-9a-f]{40}")


def is_sha(value) -> bool:
    """True only for an exact 40-character hex object name."""
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def nonblank_str(value):
    """``value`` when it is a non-blank string, else None."""
    return value if isinstance(value, str) and value.strip() else None


def _object(container: dict, key: str):
    value = container.get(key)
    return value if isinstance(value, dict) else None


def pull_request_problem(pr) -> str | None:
    """The first structural problem in a ``repos/{repo}/pulls/{n}`` answer.

    ``None`` means every field the gates read unconditionally is present and of
    the right type, so they may be dereferenced without a guard.
    """
    if not isinstance(pr, dict):
        return f"GitHub answered with {type(pr).__name__}, not a pull request object"
    head = _object(pr, "head")
    if head is None:
        return "pull request head is not an object"
    if not is_sha(head.get("sha")):
        return "pull request head.sha is not a 40-character commit sha"
    base = _object(pr, "base")
    if base is None:
        return "pull request base is not an object"
    if nonblank_str(base.get("ref")) is None:
        return "pull request base.ref is not a branch name"
    if not isinstance(pr.get("merged"), bool):
        # The one that matters most: "false" is a non-empty string, so a
        # truthiness test on it reads an unmerged PR as merged.
        return "pull request merged is not a boolean"
    if nonblank_str(pr.get("state")) is None:
        return "pull request state is not a status string"
    if pr.get("merged_at") is not None and not isinstance(pr.get("merged_at"), str):
        return "pull request merged_at is neither null nor a timestamp string"
    return None
