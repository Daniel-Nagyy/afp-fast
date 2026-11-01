"""
Pipeline orchestrator: one run per repo, everything from the GitHub API
(no cloning). batch.py runs this over a whole corpus.

  1. Eligibility first: one GraphQL query (stars, fork/archived, push
     date, commit count) and the file tree (lines of code estimated from
     file sizes). A repo that fails a rule stops here with status
     "ineligible".
  2. Everything else at once: commits in the window, PRs, issues, the
     contributor count and review rules, and the few small files the repo
     scanner reads.
  3. Only the top five agents are detected (signatures.TOP_BRANDS), on
     every channel.

Output of one run is a dict of lists of plain dicts (JSON-serializable),
written to JSONL files by export_jsonl().
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import repo_scan
from .adoption import summarize_repo
from . import config
from .collect import code_bytes, compute_edit_velocity
from .features import commits_to_feature_vectors
from .heuristics import apply_pr_label
from .schema import EligibilityResult, FileMarker, RepoAdoption, RepoMetadata
from .signatures import file_marker_brand

class PipelineResult:
    def __init__(self, full_name: str = ""):
        self.full_name = full_name
        self.repo_metadata: Optional[RepoMetadata] = None
        self.eligibility: Optional[EligibilityResult] = None
        self.commits: List[Any] = []
        self.pull_requests: List[Any] = []
        self.issues: List[Any] = []
        self.feature_vectors: List[Any] = []
        self.file_markers: List[Any] = []
        self.signals: List[Any] = []
        self.adoption: Optional[RepoAdoption] = None
        self.mode: str = "api"
        self.window: Dict[str, Optional[str]] = {"since": None, "until": None}
        self.notes: List[str] = []
        self.status: str = "ok"
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_name": self.full_name,
            "status": self.status,
            "error": self.error,
            "mode": self.mode,
            "window": self.window,
            "notes": self.notes,
            "repo_metadata": self.repo_metadata.to_dict() if self.repo_metadata else None,
            "eligibility": self.eligibility.to_dict() if self.eligibility else None,
            "adoption": self.adoption.to_dict() if self.adoption else None,
            "commit_count": len(self.commits),
            "pr_count": len(self.pull_requests),
            "issue_count": len(self.issues),
            "feature_vector_count": len(self.feature_vectors),
            "file_marker_count": len(self.file_markers),
            "signal_count": len(self.signals),
        }


def _propagate_pr_labels(commits, prs) -> int:
    """Label the commit GitHub created when merging an agent PR, and mark
    squash commits as such so edit velocity excludes them. Returns the
    number of commits that gained an agent label this way."""
    by_sha = {c.sha: c for c in commits}
    gained = 0
    for pr in prs:
        commit = by_sha.get(pr.merge_commit_sha or "")
        if commit is None:
            continue
        if pr.merged_via == "squash" and not commit.is_merge:
            commit.merge_type = "squash"
        was_agent = commit.label.label == "agent_labeled"
        apply_pr_label(commit.label, pr.label, pr.number)
        if not was_agent and commit.label.label == "agent_labeled":
            gained += 1
    return gained


def _markers_from_paths(paths: List[str]) -> List[FileMarker]:
    markers, seen = [], set()
    for rel in paths:
        hit = file_marker_brand(rel)
        if not hit:
            continue
        marker, brand = hit
        key = (marker, rel.rsplit("/", 1)[0] if marker.endswith("/") else rel)
        if key not in seen:
            seen.add(key)
            markers.append(FileMarker(path=rel, marker=marker, brand=brand))
    return markers[:50]


def run_pipeline(owner: str, repo: str, since: Optional[str] = None, until: Optional[str] = None,
                 client=None, collect_prs: bool = True, max_prs: Optional[int] = None,
                 as_of: Optional[str] = None, collect_issues: bool = True,
                 max_issues: int = 200, max_commits: int = 2000) -> PipelineResult:
    """Run every stage for one repo. `since`/`until` (YYYY-MM-DD) bound the
    commit, PR and issue window; `as_of` (YYYY-MM-DD, default today) is the
    date "pushed in the last 12 months" is measured from; `client` lets a
    batch share one rate-limited GitHubClient (and its token pool) across
    workers (default: built from GITHUB_TOKEN(S))."""
    full_name = f"{owner}/{repo}"
    result = PipelineResult(full_name)
    result.window = {"since": since, "until": until}
    if client is None:
        from .github_client import GitHubClient, tokens_from_env
        if not tokens_from_env():
            raise ValueError("set GITHUB_TOKEN or GITHUB_TOKENS")
        client = GitHubClient()
    now = datetime.fromisoformat(f"{as_of}T23:59:59+00:00") if as_of else None

    with ThreadPoolExecutor(max_workers=6) as pool:
        return _run(result, owner, repo, since, until, client, collect_prs, max_prs, collect_issues,
                    max_issues, max_commits, now, pool)


def _run(result, owner, repo, since, until, client, collect_prs, max_prs, collect_issues, max_issues,
         max_commits, now, pool) -> PipelineResult:
    from .api_collect import collect_commits_api
    from .eligibility import _size_tier, build_repo_metadata, check_eligibility
    from .issue_collect import collect_issues as _collect_issues
    from .pr_collect import collect_pull_requests
    tree_future = pool.submit(client.get_tree, owner, repo, "HEAD")

    # --- 1. eligibility first ---------------------------------------------
    meta = build_repo_metadata(client, owner, repo, fetch_contributors=False, fetch_branch_protection=False)
    entries, truncated = tree_future.result()
    if truncated:
        result.notes.append("file tree truncated by GitHub (very large repo); some files may be missed")
    else:
        meta.code_bytes = code_bytes([(e["path"], e.get("size") or 0) for e in entries])
        meta.total_loc, meta.loc_estimated = round(meta.code_bytes / config.BYTES_PER_LOC), True
        meta.size_tier = _size_tier(meta.total_loc)
    result.repo_metadata = meta
    result.eligibility = check_eligibility(meta, now=now)
    if result.eligibility.eligible is False:
        result.status = "ineligible"
        return result

    # --- 2. everything else, side by side ----------------------------------
    paths = [e["path"] for e in entries]
    futures: Dict[str, Any] = {
        "meta": pool.submit(build_repo_metadata, client, owner, repo, base=meta),
        "commits": pool.submit(collect_commits_api, client, owner, repo, since=since, until=until,
                               max_commits=max_commits),
        "texts": pool.submit(client.get_blobs, owner, repo, repo_scan.wanted_paths(paths)),
    }
    if collect_prs:
        futures["prs"] = pool.submit(collect_pull_requests, client, owner, repo,
                                     since=since, until=until, max_prs=max_prs)
    if collect_issues:
        futures["issues"] = pool.submit(_collect_issues, client, owner, repo,
                                        since=since, until=until, max_issues=max_issues)
    result.file_markers = _markers_from_paths(paths)
    result.signals = repo_scan.scan(paths, futures["texts"].result().get)
    result.commits = commits = futures["commits"].result()
    result.repo_metadata = futures["meta"].result()
    if "prs" in futures:
        result.pull_requests = futures["prs"].result()
        gained = _propagate_pr_labels(commits, result.pull_requests)
        if gained:
            result.notes.append(f"{gained} merge/squash commit(s) labelled from their agent PR")
    if "issues" in futures:
        result.issues = futures["issues"].result()

    # --- 3. features and adoption (after PR propagation, so squash
    # commits are known before velocity is ranked) ------------------------
    compute_edit_velocity(commits)
    result.feature_vectors = commits_to_feature_vectors(commits)
    result.adoption = summarize_repo(result.full_name, commits, result.pull_requests, result.file_markers,
                                     signals=result.signals, issues=result.issues)
    return result


def _records_to_dicts(records: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        out.append(r.to_dict() if hasattr(r, "to_dict") else asdict(r))
    return out


def export_jsonl(result: PipelineResult, out_dir: str) -> Dict[str, str]:
    """Write commits/prs/issues/features/file_markers/signals .jsonl and
    summary.json (UTF-8) to out_dir. Returns the paths written.
    summary.json is written last: its presence marks the repo as done."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _write_jsonl(name: str, records: List[Any]):
        path = out / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for rec_dict in _records_to_dicts(records):
                f.write(json.dumps(rec_dict, default=str, ensure_ascii=False) + "\n")
        paths[name] = str(path)

    _write_jsonl("commits.jsonl", result.commits)
    _write_jsonl("prs.jsonl", result.pull_requests)
    _write_jsonl("issues.jsonl", result.issues)
    _write_jsonl("features.jsonl", result.feature_vectors)
    _write_jsonl("file_markers.jsonl", result.file_markers)
    _write_jsonl("signals.jsonl", result.signals)

    summary_path = out / "summary.json"
    tmp_path = out / "summary.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str, ensure_ascii=False)
    os.replace(tmp_path, summary_path)
    paths["summary.json"] = str(summary_path)

    return paths
