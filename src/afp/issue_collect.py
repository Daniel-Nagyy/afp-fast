"""
Issue-level agent traces via GraphQL: issues assigned to an agent (the
issue is the agent's task), agents @mentioned in the issue or its
comments (`@claude`, `@codex`, `@copilot` ...), and comments written by
agent bots. One request returns a page of 50 issues with up to 20
comments each.

A mention alone can be people talking about a tool; an agent bot
commenting on the same issue confirms it was used (see
IssueRecord.agent_commenter_brands).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from . import signatures as sig_db
from .github_client import GitHubClient
from .schema import IssueRecord

ISSUE_PAGE_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $n: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $n, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number createdAt body
        author { login __typename }
        assignedActors(first: 5) { nodes { __typename ... on Bot { login } ... on User { login } } }
        comments(first: 20) { nodes { body author { login __typename } } }
      }
    }
  }
}
"""


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def login_of(actor: Optional[dict]) -> Optional[str]:
    """GraphQL bot logins come without the REST "[bot]" suffix."""
    if not actor:
        return None
    login = actor.get("login")
    if login and actor.get("__typename") == "Bot" and not login.endswith("[bot]"):
        login += "[bot]"
    return login


def _add(target: List[str], brand: Optional[str]) -> None:
    if brand and brand not in target:
        target.append(brand)


def record_from_node(owner: str, repo: str, node: dict) -> IssueRecord:
    author = login_of(node.get("author"))
    rec = IssueRecord(repo_full_name=f"{owner}/{repo}", number=node["number"], created_at=node.get("createdAt"),
                      author_login=author, author_brand=sig_db.brand_for_login(author))
    for actor in (node.get("assignedActors") or {}).get("nodes") or []:
        _add(rec.assignee_brands, sig_db.brand_for_login(login_of(actor)))
    for brand in sig_db.mentioned_brands(node.get("body")):
        _add(rec.mention_brands, brand)
    for c in (node.get("comments") or {}).get("nodes") or []:
        for brand in sig_db.mentioned_brands(c.get("body")):
            _add(rec.mention_brands, brand)
        _add(rec.agent_commenter_brands, sig_db.brand_for_login(login_of(c.get("author"))))
    return rec


def collect_issues(client: GitHubClient, owner: str, repo: str, since: Optional[str] = None,
                   until: Optional[str] = None, max_issues: int = 200, page_size: int = 50) -> List[IssueRecord]:
    """Newest-first issues created in [since, until], up to `max_issues`
    (0 = no cap). Only issues with some agent involvement are returned."""
    since_dt = _parse_dt(f"{since}T00:00:00Z") if since else None
    until_dt = _parse_dt(f"{until}T23:59:59Z") if until else None
    found: List[IssueRecord] = []
    seen = 0
    variables = {"owner": owner, "name": repo, "cursor": None, "n": page_size}
    while True:
        data = client.graphql_page(ISSUE_PAGE_QUERY, variables)
        repo_data = data.get("repository")
        if repo_data is None:
            break
        conn = repo_data["issues"]
        stop = False
        for node in conn.get("nodes") or []:
            if not node:
                continue
            created = _parse_dt(node.get("createdAt"))
            if until_dt and created and created > until_dt:
                continue
            if since_dt and created and created < since_dt:
                stop = True
                break
            seen += 1
            rec = record_from_node(owner, repo, node)
            if rec.author_brand or rec.assignee_brands or rec.mention_brands or rec.agent_commenter_brands:
                found.append(rec)
            if max_issues and seen >= max_issues:
                stop = True
                break
        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        variables["cursor"] = conn["pageInfo"]["endCursor"]
    return found
