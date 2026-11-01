"""
Repo eligibility + metadata checks, per the "Repo eligibility and time
window" table in Shared_Corpus_Methodology.tex.

build_repo_metadata() gathers a RepoMetadata record in three requests:
one GraphQL query (stars, fork/archived, push date, languages, default
branch and its full commit count), one REST call for the contributor
count, and the publicly readable rulesets endpoint for "review
required". Lines of code are estimated by the pipeline from file sizes in
the tree listing (collect.code_bytes). check_eligibility() applies the group's thresholds and returns
a pass/fail with reasons, so a rejected repo's reasons can be logged
rather than silently dropped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from . import config
from .github_client import GitHubClient
from .schema import RepoMetadata, EligibilityResult

REPO_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner stargazerCount isFork isArchived pushedAt createdAt
    primaryLanguage { name }
    languages(first: 30, orderBy: {field: SIZE, direction: DESC}) { edges { size node { name } } }
    defaultBranchRef { name target { ... on Commit { history { totalCount } } } }
  }
}
"""


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


def _size_tier(total_loc: Optional[int]) -> Optional[str]:
    if total_loc is None:
        return None
    for tier, (lo, hi) in config.SIZE_TIER_BOUNDARIES.items():
        if total_loc >= lo and (hi is None or total_loc < hi):
            return tier
    return None


def build_repo_metadata(client: GitHubClient, owner: str, repo: str,
                        fetch_contributors: bool = True,
                        fetch_branch_protection: bool = True,
                        total_loc: Optional[int] = None,
                        base: Optional[RepoMetadata] = None) -> RepoMetadata:
    """`base` is metadata already fetched without the REST lookups
    (--eligible-only); only the REST lookups are added to it. The three
    requests are independent of each other and run side by side."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        contributors = pool.submit(client.get_contributors_count, owner, repo) if fetch_contributors else None
        meta = base or _query_metadata(client, owner, repo, total_loc)
        review = pool.submit(client.review_required, owner, repo, meta.default_branch) \
            if fetch_branch_protection else None
    if contributors is not None:
        try:
            meta.contributor_count = contributors.result()
            meta.is_solo = meta.contributor_count == 1
        except Exception:
            meta.contributor_count = None
    if review is not None:
        try:
            meta.branch_protection, meta.branch_protection_source = review.result()
        except Exception:
            meta.branch_protection = None
    return meta


def _query_metadata(client: GitHubClient, owner: str, repo: str, total_loc: Optional[int]) -> RepoMetadata:
    data = client.graphql(REPO_QUERY, {"owner": owner, "name": repo})
    raw = data.get("repository")
    if raw is None:
        raise LookupError(f"{owner}/{repo} not found (deleted, renamed away or private)")

    languages = {e["node"]["name"]: e["size"] for e in (raw.get("languages") or {}).get("edges") or []}
    branch = raw.get("defaultBranchRef") or {}
    history = ((branch.get("target") or {}).get("history") or {})

    meta = RepoMetadata(
        owner=owner,
        name=repo,
        full_name=raw.get("nameWithOwner") or f"{owner}/{repo}",
        stars=raw.get("stargazerCount") or 0,
        is_fork=bool(raw.get("isFork")),
        is_archived=bool(raw.get("isArchived")),
        pushed_at=raw.get("pushedAt"),
        created_at=raw.get("createdAt"),
        primary_language=(raw.get("primaryLanguage") or {}).get("name"),
        languages=languages,
        language_bytes=sum(languages.values()) if languages else None,
        total_loc=total_loc,
        commit_count=history.get("totalCount"),
        commit_count_source="graphql" if history.get("totalCount") is not None else None,
        default_branch=branch.get("name") or "main",
    )
    meta.stack_tag = config.LANGUAGE_TO_STACK_TAG.get(meta.primary_language or "", meta.primary_language)
    meta.size_tier = _size_tier(meta.total_loc)
    return meta


def check_eligibility(meta: RepoMetadata, commit_count: Optional[int] = None,
                      now: Optional[datetime] = None) -> EligibilityResult:
    """Apply the group's eligibility thresholds. `commit_count` overrides
    meta.commit_count when given; it must be the repo's full history,
    not a window count."""
    now = now or datetime.now(timezone.utc)
    reasons = []

    if meta.is_fork and not config.ALLOW_FORKS:
        reasons.append("is a fork")

    if meta.is_archived:
        reasons.append("is archived")

    pushed_dt = _parse_dt(meta.pushed_at)
    if pushed_dt is None:
        reasons.append("pushed_at missing/unparseable")
    else:
        age_days = (now - pushed_dt).days
        if age_days > config.PUSH_RECENCY_MONTHS * 30:
            reasons.append(
                f"not pushed in last {config.PUSH_RECENCY_MONTHS} months "
                f"(last push {age_days} days ago)"
            )

    if meta.stars < config.MIN_STARS:
        reasons.append(f"stars {meta.stars} < minimum {config.MIN_STARS}")

    unchecked = []
    if meta.total_loc is not None and meta.total_loc < config.MIN_LOC:
        reasons.append(f"total_loc {meta.total_loc} < minimum {config.MIN_LOC}")
    elif meta.total_loc is None:
        unchecked.append("total_loc (file tree too large to list)")

    effective_commit_count = commit_count if commit_count is not None else meta.commit_count
    if effective_commit_count is not None and effective_commit_count < config.MIN_COMMITS:
        reasons.append(f"commit_count {effective_commit_count} < minimum {config.MIN_COMMITS}")
    elif effective_commit_count is None:
        unchecked.append("commit_count (unknown)")

    if getattr(meta, "loc_estimated", False) and meta.total_loc is not None:
        unchecked.append(f"total_loc estimated from file sizes (~{meta.total_loc} lines)")

    return EligibilityResult(
        full_name=meta.full_name,
        eligible=False if reasons else (None if unchecked else True),
        reasons=reasons,
        unchecked=unchecked,
        checked_at=now.isoformat(),
    )
