"""
Shared pytest fixtures.

synthetic_commits builds six commits from GraphQL history nodes (what
GitHub's API returns, newest first) with known ground truth:
  1. Human commit (plain)
  2. Human commit (plain, 1h later)
  3. Claude Code trailer commit, 500 lines added 60s after commit 2 -> should
     be label=agent_labeled/claude_code/trailer AND edit_velocity_flagged=True
  4. codex-bot e-mail commit (noreply@openai.com), 300 lines, no trailer ->
     label=agent_labeled/codex/email_domain
  5. Lockfile-update commit, 400 lines, 120s after commit 3 (the API gives
     no changed paths, so it is ranked like any other commit)
  6. Human commit the next day (normal pace) -> label=unlabeled
"""

import pytest

from afp import signatures as sig_db
from afp.api_collect import commit_from_node

HUMAN = ("Human Dev", "human.dev@example.com")


def _node(oid, message, date, additions, author=HUMAN):
    name, email = author
    return {"oid": oid, "message": message, "authoredDate": date, "committedDate": date,
            "additions": additions, "deletions": 0, "changedFilesIfAvailable": 1,
            "parents": {"nodes": [{"oid": f"p{oid}"}], "totalCount": 1},
            "author": {"name": name, "email": email, "date": date, "user": None},
            "committer": {"name": name, "email": email, "date": date}}


@pytest.fixture
def synthetic_commits():
    nodes = [
        _node("c6", "docs: add notes", "2026-03-02T11:03:00+00:00", 1),
        _node("c5", "chore: update lockfile", "2026-03-01T11:03:00+00:00", 400),
        _node("c4", "Generate module", "2026-03-01T11:02:00+00:00", 300,
              author=("codex-bot", "noreply@openai.com")),
        _node("c3", "Implement large feature\n\nCo-authored-by: Claude <noreply@anthropic.com>",
              "2026-03-01T11:01:00+00:00", 500),
        _node("c2", "Add world print", "2026-03-01T11:00:00+00:00", 1),
        _node("c1", "Initial commit", "2026-03-01T10:00:00+00:00", 1),
    ]
    return [commit_from_node("test/synthetic", n) for n in nodes]


@pytest.fixture
def all_agents():
    """Detect every agent in signatures.py (the tool itself detects only
    the top five), for tests of the full signature tables."""
    sig_db.set_active_brands(None)
    yield
    sig_db.set_active_brands(sig_db.TOP_BRANDS)
