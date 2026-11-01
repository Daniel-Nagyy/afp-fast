"""A GitHubClient that answers every request the pipeline makes for one
repo, offline: an eligible repo with one Claude Code commit, CLAUDE.md
and a workflow running the Codex action."""

from afp.github_client import GitHubClient


class FakeAPI(GitHubClient):
    def __init__(self, token="t", **kw):
        super().__init__(token=token, sleep=lambda s: None)

    def graphql(self, query, variables=None):
        if "stargazerCount" in query:
            return {"repository": {"nameWithOwner": "o/r", "stargazerCount": 50, "isFork": False, "isArchived": False,
                                   "pushedAt": "2026-06-01T00:00:00Z", "createdAt": "2020-01-01T00:00:00Z",
                                   "primaryLanguage": {"name": "Python"}, "languages": {"edges": []},
                                   "defaultBranchRef": {"name": "main", "target": {"history": {"totalCount": 500}}}}}
        if "history(first: $n" in query:
            node = {"oid": "c1", "message": "Fix\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
                    "authoredDate": "2026-03-01T10:00:00Z", "committedDate": "2026-03-01T10:00:00Z",
                    "additions": 10, "deletions": 1, "changedFilesIfAvailable": 1,
                    "parents": {"nodes": [{"oid": "c0"}], "totalCount": 1},
                    "author": {"name": "Alice", "email": "alice@example.com", "user": {"login": "alice"}},
                    "committer": {"name": "Alice", "email": "alice@example.com"}}
            return {"repository": {"defaultBranchRef": {"target": {"history": {
                "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [node]}}}}}
        if "pullRequests(" in query or "issues(" in query:
            key = "pullRequests" if "pullRequests(" in query else "issues"
            return {"repository": {key: {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}}}
        if "object(expression" in query:
            return {"repository": {"f0": {"byteSize": 60, "isBinary": False,
                                          "text": "steps:\n  - uses: openai/codex-action@v1\n"}}}
        raise AssertionError(query[:80])

    def get_tree(self, owner, repo, ref):
        return [{"path": "CLAUDE.md", "type": "blob", "size": 900},
                {"path": "src/app.py", "type": "blob", "size": 320_000},
                {"path": ".github/workflows/codex.yml", "type": "blob", "size": 60}], False

    def get_contributors_count(self, owner, repo):
        return 7

    def review_required(self, owner, repo, branch):
        return None, None
