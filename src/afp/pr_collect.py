"""
Pull-request-level data collection, per the "Pull request-level feature
parameters" table.

Two paths produce the same PullRequestRecord:

  - collect_pull_requests() -- GraphQL, used by the pipeline. One request
    returns a page of 50 PRs with their reviews, labels, merge commit and
    commenters, so a repo's newest 300 PRs cost ~6 requests instead of
    the ~900 the per-PR REST path needs. Pages stop once PRs are older
    than the collection window.
  - build_pr_record() -- REST, one PR at a time (kept for spot checks and
    the mocked tests).

Review timing ignores reviews by the PR's own author and by bots: an AI
reviewer answering within seconds is not a human "first review", and
counting it would make every bot-reviewed PR look rubber-stamped. AI
review bots are recorded separately in `ai_reviewers`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config
from . import signatures as sig_db
from .github_client import GitHubClient
from .heuristics import resolve_label
from .schema import PullRequestRecord


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _merged_via(pr_raw: dict) -> Optional[str]:
    """GitHub's REST PR object doesn't say how a PR was merged. The
    GraphQL path infers it (see _infer_merged_via); a REST caller may set
    `_merged_via_hint`. Otherwise this stays None rather than guessing."""
    if not pr_raw.get("merged_at"):
        return None
    return pr_raw.get("_merged_via_hint")


def _infer_merged_via(number: int, merge_commit: Optional[dict], pr_commit_count: Optional[int]) -> Optional[str]:
    """Merge method from the commit GitHub created on the base branch:
    two parents = merge commit; one parent and GitHub's default squash
    title "... (#N)" = squash; one parent otherwise = rebase. A one-commit
    PR merged with one parent cannot be told apart."""
    if not merge_commit:
        return None
    parents = (merge_commit.get("parents") or {}).get("totalCount")
    if parents is None:
        return None
    if parents > 1:
        return "merge"
    headline = merge_commit.get("messageHeadline") or ""
    if headline.rstrip().endswith(f"(#{number})"):
        return "squash"
    if pr_commit_count == 1:
        return "squash_or_rebase"
    return "rebase"


def _normalise_login(author: Optional[dict]) -> Optional[str]:
    """GraphQL reports bot logins without the REST "[bot]" suffix."""
    if not author:
        return None
    login = author.get("login")
    if login and author.get("__typename") == "Bot" and not login.endswith("[bot]"):
        login += "[bot]"
    return login


def _is_bot(login: Optional[str]) -> bool:
    if not login:
        return False
    low = login.lower()
    return low.endswith("[bot]") or low in sig_db.AGENT_LOGINS or low in sig_db.REVIEW_BOT_LOGINS \
        or low in sig_db.CI_BOT_LOGINS


def _review_timing(created_at: Optional[str], author_login: Optional[str],
                   reviews: Iterable[dict]) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    """(first_review_at, latency, first_approval_at, approval_latency) over
    reviews by someone other than the PR author and not a bot."""
    created_dt = _parse_dt(created_at)
    human: List[Tuple[datetime, str]] = []
    for r in reviews:
        login = r.get("_login")
        if login is None:
            login = (r.get("user") or r.get("author") or {}).get("login")
        if _is_bot(login) or (author_login and login and login.lower() == author_login.lower()):
            continue
        dt = _parse_dt(r.get("submitted_at") or r.get("submittedAt"))
        if dt is not None:
            human.append((dt, (r.get("state") or "").upper()))
    if not human:
        return None, None, None, None
    first = min(dt for dt, _ in human)
    approvals = [dt for dt, state in human if state == "APPROVED"]
    first_approval = min(approvals) if approvals else None

    def _latency(dt: Optional[datetime]) -> Optional[float]:
        return (dt - created_dt).total_seconds() if dt and created_dt else None

    return (first.isoformat(), _latency(first),
            first_approval.isoformat() if first_approval else None, _latency(first_approval))


def _finish_record(record: PullRequestRecord, reviews: List[dict], reviewer_logins: Iterable[str]) -> PullRequestRecord:
    record.pr_size = record.additions + record.deletions
    record.review_depth = (record.review_comment_count / record.pr_size) if record.pr_size > 0 else None
    (record.first_review_at, record.review_latency_seconds,
     record.first_approval_at, record.approval_latency_seconds) = _review_timing(
        record.created_at, record.author_login, reviews)
    record.rubber_stamp_flag = (
        record.review_comment_count <= config.RUBBER_STAMP_MAX_COMMENTS
        and record.approval_latency_seconds is not None
        and record.approval_latency_seconds <= config.RUBBER_STAMP_MAX_LATENCY_SECONDS
    )
    ai = []
    for login in reviewer_logins:
        name = sig_db.REVIEW_BOT_LOGINS.get((login or "").lower())
        if name and name not in ai:
            ai.append(name)
    record.ai_reviewers = ai
    record.label = resolve_label(
        author_login=record.author_login,
        pr_title=record.title,
        pr_body=record.body or "",
        pr_branch_name=record.branch_name or "",
        pr_labels=record.labels,
    )
    return record


# ---------------------------------------------------------------------------
# REST, one PR at a time
# ---------------------------------------------------------------------------

def build_pr_record(client: GitHubClient, owner: str, repo: str, number: int) -> PullRequestRecord:
    pr_raw = client.get_pull_request(owner, repo, number)
    reviews = client.list_pr_reviews(owner, repo, number)
    review_comments = client.list_pr_review_comments(owner, repo, number)

    head = pr_raw.get("head", {}) or {}
    record = PullRequestRecord(
        repo_full_name=f"{owner}/{repo}",
        number=number,
        title=pr_raw.get("title", "") or "",
        body=pr_raw.get("body"),
        branch_name=head.get("ref"),
        labels=[lbl.get("name", "") for lbl in (pr_raw.get("labels") or [])],
        author_login=(pr_raw.get("user") or {}).get("login"),
        created_at=pr_raw.get("created_at"),
        merged_at=pr_raw.get("merged_at"),
        merge_commit_sha=pr_raw.get("merge_commit_sha"),
        merged_via=_merged_via(pr_raw),
        commit_count=pr_raw.get("commits"),
        additions=pr_raw.get("additions", 0) or 0,
        deletions=pr_raw.get("deletions", 0) or 0,
        changed_files=pr_raw.get("changed_files", 0) or 0,
        review_comment_count=len(review_comments),
    )
    reviewers = [(r.get("user") or {}).get("login") for r in reviews]
    reviewers += [(c.get("user") or {}).get("login") for c in review_comments]
    return _finish_record(record, reviews, reviewers)


def collect_pull_requests_rest(client: GitHubClient, owner: str, repo: str,
                               state: str = "closed", max_pages: int = 10,
                               max_prs: Optional[int] = None) -> List[PullRequestRecord]:
    listing = client.list_pull_requests(owner, repo, state=state, max_pages=max_pages)
    if max_prs:
        listing = listing[:max_prs]
    return [build_pr_record(client, owner, repo, pr["number"]) for pr in listing]


# ---------------------------------------------------------------------------
# GraphQL, a page of PRs per request
# ---------------------------------------------------------------------------

PR_PAGE_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $n: Int!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    pullRequests(first: $n, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body headRefName state createdAt mergedAt
        additions deletions changedFiles
        author { login __typename }
        labels(first: 20) { nodes { name } }
        mergeCommit { oid messageHeadline parents(first: 2) { totalCount } }
        commits { totalCount }
        reviewThreads(first: 1) { totalCount }
        reviews(first: 30) {
          nodes { state submittedAt author { login __typename } }
        }
        comments(first: 20) { nodes { body author { login __typename } } }
      }
    }
  }
}
"""


def record_from_graphql(owner: str, repo: str, node: Dict[str, Any]) -> PullRequestRecord:
    reviews = []
    for r in (node.get("reviews") or {}).get("nodes") or []:
        reviews.append({"_login": _normalise_login(r.get("author")), "state": r.get("state"),
                        "submittedAt": r.get("submittedAt")})
    comments = (node.get("comments") or {}).get("nodes") or []
    commenters = [_normalise_login(c.get("author")) for c in comments]
    merge_commit = node.get("mergeCommit")
    commit_count = (node.get("commits") or {}).get("totalCount")
    record = PullRequestRecord(
        repo_full_name=f"{owner}/{repo}",
        number=node["number"],
        title=node.get("title") or "",
        body=node.get("body"),
        branch_name=node.get("headRefName"),
        labels=[lbl["name"] for lbl in (node.get("labels") or {}).get("nodes") or [] if lbl],
        author_login=_normalise_login(node.get("author")),
        created_at=node.get("createdAt"),
        merged_at=node.get("mergedAt"),
        merge_commit_sha=(merge_commit or {}).get("oid") if node.get("mergedAt") else None,
        merged_via=_infer_merged_via(node["number"], merge_commit, commit_count) if node.get("mergedAt") else None,
        commit_count=commit_count,
        additions=node.get("additions") or 0,
        deletions=node.get("deletions") or 0,
        changed_files=node.get("changedFiles") or 0,
        # review threads: one per inline comment chain; the GraphQL-cheap
        # stand-in for REST's review-comment count
        review_comment_count=(node.get("reviewThreads") or {}).get("totalCount") or 0,
    )
    for text in [node.get("body")] + [c.get("body") for c in comments]:
        for brand in sig_db.mentioned_brands(text):
            if brand not in record.mention_brands:
                record.mention_brands.append(brand)
    for login in commenters + [r["_login"] for r in reviews]:
        # AI review bots reviewing a PR are review adoption (ai_reviewers),
        # not use of a coding agent on the PR
        if not login or login.lower() in sig_db.REVIEW_BOT_LOGINS:
            continue
        brand = sig_db.brand_for_login(login)
        if brand and login.lower() != (record.author_login or "").lower() \
                and brand not in record.agent_commenter_brands:
            record.agent_commenter_brands.append(brand)
    return _finish_record(record, reviews, [r["_login"] for r in reviews] + commenters)


def collect_pull_requests(client: GitHubClient, owner: str, repo: str,
                          since: Optional[str] = None, until: Optional[str] = None,
                          max_prs: Optional[int] = None,
                          page_size: int = config.PR_PAGE_SIZE) -> List[PullRequestRecord]:
    """Newest-first PRs created in [since, until], up to `max_prs`
    (config.MAX_PRS_PER_REPO by default; 0 = no cap)."""
    if max_prs is None:
        max_prs = config.MAX_PRS_PER_REPO
    since_dt = _parse_dt(f"{since}T00:00:00Z") if since else None
    until_dt = _parse_dt(f"{until}T23:59:59Z") if until else None
    records: List[PullRequestRecord] = []
    variables = {"owner": owner, "name": repo, "cursor": None, "n": page_size}
    while True:
        data = client.graphql_page(PR_PAGE_QUERY, variables)
        repo_data = data.get("repository")
        if repo_data is None:
            break
        conn = repo_data["pullRequests"]
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
            records.append(record_from_graphql(owner, repo, node))
            if max_prs and len(records) >= max_prs:
                stop = True
                break
        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        variables["cursor"] = conn["pageInfo"]["endCursor"]
    return records
