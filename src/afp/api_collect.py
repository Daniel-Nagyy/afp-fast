"""
Commit collection from GitHub's GraphQL history (no cloning).

One request returns 100 commits with their full message, author and
committer identity, the author's GitHub login (which git log does not
have, so bot authors are matched exactly), additions, deletions, the
number of files changed and the parent count. What it cannot give is
the list of changed paths, so the file-type mix features stay 0 and LOC
is estimated from file sizes; everything used for agent detection and
edit velocity is available.
"""

from __future__ import annotations

from typing import List, Optional

from .github_client import GitHubClient
from .heuristics import resolve_label
from .schema import CommitRecord

HISTORY_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $since: GitTimestamp, $until: GitTimestamp, $n: Int!) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef { target { ... on Commit {
      history(first: $n, after: $cursor, since: $since, until: $until) {
        pageInfo { hasNextPage endCursor }
        nodes {
          oid message authoredDate committedDate additions deletions changedFilesIfAvailable
          parents(first: 2) { nodes { oid } totalCount }
          author { name email date user { login } }
          committer { name email date }
        }
      }
    } } }
  }
}
"""


def commit_from_node(full_name: str, node: dict) -> CommitRecord:
    author = node.get("author") or {}
    committer = node.get("committer") or {}
    parents = [p["oid"] for p in (node.get("parents") or {}).get("nodes") or []]
    is_merge = ((node.get("parents") or {}).get("totalCount") or 0) > 1
    login = (author.get("user") or {}).get("login")
    commit = CommitRecord(
        repo_full_name=full_name,
        sha=node["oid"],
        author_name=author.get("name") or None,
        author_email=author.get("email") or None,
        committer_name=committer.get("name") or None,
        committer_email=committer.get("email") or None,
        # author.date / committer.date keep the original timezone offset
        # (authoredDate is converted to UTC), so hour-of-day features
        # reflect the author's local time, as git shows it
        author_date=author.get("date") or node.get("authoredDate"),
        commit_date=committer.get("date") or node.get("committedDate"),
        message=(node.get("message") or "").rstrip(),
        parents=parents,
        is_merge=is_merge,
        merge_type="merge" if is_merge else None,
        additions=node.get("additions") or 0,
        deletions=node.get("deletions") or 0,
        files_changed=node.get("changedFilesIfAvailable") or 0,
    )
    commit.label = resolve_label(
        commit_message=commit.message,
        author_login=login,
        author_email=commit.author_email,
        committer_email=commit.committer_email,
        author_name=commit.author_name,
        committer_name=commit.committer_name,
    )
    return commit


def collect_commits_api(client: GitHubClient, owner: str, repo: str, since: Optional[str] = None,
                        until: Optional[str] = None, max_commits: int = 2000) -> List[CommitRecord]:
    """Default-branch commits in [since, until] (newest first), up to
    `max_commits` (0 = all)."""
    full_name = f"{owner}/{repo}"
    variables = {"owner": owner, "name": repo, "cursor": None,
                 "since": f"{since}T00:00:00Z" if since else None,
                 "until": f"{until}T23:59:59Z" if until else None, "n": 100}
    commits: List[CommitRecord] = []
    while True:
        data = client.graphql_page(HISTORY_QUERY, variables)
        target = (((data.get("repository") or {}).get("defaultBranchRef") or {}).get("target") or {})
        history = target.get("history")
        if not history:
            break
        for node in history.get("nodes") or []:
            commits.append(commit_from_node(full_name, node))
            if max_commits and len(commits) >= max_commits:
                return commits
        if not history["pageInfo"]["hasNextPage"]:
            break
        variables["cursor"] = history["pageInfo"]["endCursor"]
    return commits
