"""
Corpus-scale runs: the per-repo pipeline over a list of repos, in
parallel, resumable.

  - Input is a gitcorpus export (corpus.csv / repos CSV with a
    `full_name` column), any CSV with `full_name`/`repo`/`nameWithOwner`,
    or a plain text file with one owner/repo per line.
  - Each repo writes to <out>/repos/<owner>__<repo>/; its summary.json is
    written last, so a repo with a summary is finished. Re-running the
    same command skips finished repos (and, unless --retry-failed,
    repos that failed), so an interrupted or crashed run resumes where
    it stopped.
  - Workers share one GitHubClient: rate limits are respected globally
    and a sleep for the reset is logged rather than failing repos.
  - A repo that fails an eligibility rule is saved with status
    "ineligible" after one query and the file tree, and counts as done.
  - When every repo is processed the outputs are merged into corpus
    files (see adoption.merge_corpus).

API cost per repo in full mode: 1 GraphQL query for metadata, 1 per 50
PRs (6 for the default 300-PR cap), and 3 REST calls (contributors,
rulesets, classic protection). With one token that is roughly 1,500
repos/hour on the REST budget; cloning is usually the real bottleneck.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from .adoption import format_report, merge_corpus
from .pipeline import PipelineResult, export_jsonl, run_pipeline

log = logging.getLogger("afp.batch")

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_NAME_COLUMNS = ("full_name", "repo", "repository", "nameWithOwner", "name_with_owner")


def read_repo_list(path: str) -> List[str]:
    """owner/repo names from a CSV (first matching name column) or a text
    file, de-duplicated in input order. github.com URLs are accepted."""
    names: List[str] = []
    p = Path(path)
    with open(p, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        if "," in head or p.suffix.lower() == ".csv":
            reader = csv.DictReader(f)
            column = next((c for c in _NAME_COLUMNS if c in (reader.fieldnames or [])), None)
            if column is None:
                raise ValueError(f"{path}: no repo column (expected one of {', '.join(_NAME_COLUMNS)})")
            raw = [row.get(column) or "" for row in reader]
        else:
            raw = [line for line in f]
    seen = set()
    for item in raw:
        item = item.strip()
        if not item or item.startswith("#"):
            continue
        item = re.sub(r"^https?://github\.com/", "", item).rstrip("/")
        if item.endswith(".git"):
            item = item[:-4]
        if not _NAME_RE.match(item):
            log.warning("skipping %r: not owner/repo", item)
            continue
        if item.lower() not in seen:
            seen.add(item.lower())
            names.append(item)
    return names


def repo_dir(out_dir: str, full_name: str) -> Path:
    owner, repo = full_name.split("/", 1)
    return Path(out_dir) / "repos" / f"{owner}__{repo}"


def _status(out_dir: str, full_name: str) -> Optional[str]:
    s = repo_dir(out_dir, full_name) / "summary.json"
    if not s.exists():
        return None
    try:
        with open(s, encoding="utf-8") as f:
            return json.load(f).get("status")
    except (OSError, ValueError):
        return None


def _write_failure(out_dir: str, full_name: str, exc: BaseException, window: Dict) -> None:
    result = PipelineResult(full_name)
    result.status = "failed"
    result.error = f"{exc.__class__.__name__}: {exc}"[:2000]
    result.window = window
    export_jsonl(result, str(repo_dir(out_dir, full_name)))


def _one_line(result: PipelineResult) -> str:
    ad = result.adoption
    agents = ""
    if ad and ad.brands:
        parts = []
        for b in ad.brands:
            n = ad.commit_brand_counts.get(b, 0) + ad.pr_brand_counts.get(b, 0)
            parts.append(f"{b} ({n})" if n else f"{b} (repo)")
        agents = "; agents: " + ", ".join(parts)
        if ad.hidden_use:
            agents += " [hidden use]"
    elig = ""
    if result.status == "ineligible":
        return "skipped, not eligible (" + "; ".join(result.eligibility.reasons) + ")"
    if result.eligibility is not None:
        elig = {True: "; eligible", False: "; not eligible"}.get(result.eligibility.eligible,
                                                                    "; eligible (LOC estimated)")
    return (f"{len(result.commits)} commits, {len(result.pull_requests)} PRs, {len(result.issues)} agent issues"
            f"{elig}{agents}")


def run_batch(repos: List[str], out_dir: str, *, workers: Optional[int] = None,
              since: Optional[str] = None, until: Optional[str] = None,
              collect_prs: bool = True, max_prs: Optional[int] = None,
              retry_failed: bool = False, limit: Optional[int] = None,
              as_of: Optional[str] = None, collect_issues: bool = True,
              max_issues: int = 200, max_commits: int = 2000) -> Dict:
    from .signatures import TOP_BRANDS
    if workers is None:
        workers = 24
    out = Path(out_dir)
    (out / "repos").mkdir(parents=True, exist_ok=True)

    todo = []
    skipped_done = skipped_failed = 0
    for name in repos:
        status = _status(out_dir, name)
        if status in ("ok", "ineligible"):
            skipped_done += 1
        elif status == "failed" and not retry_failed:
            skipped_failed += 1
        else:
            todo.append(name)
    if limit:
        todo = todo[:limit]

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"since": since, "until": until, "as_of": as_of, "collect_prs": collect_prs,
                   "max_prs": max_prs, "collect_issues": collect_issues, "max_issues": max_issues,
                   "workers": workers, "repos_in_list": len(repos), "agents": list(TOP_BRANDS),
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)

    from .github_client import GitHubClient, tokens_from_env
    tokens = tokens_from_env()
    if not tokens:
        raise ValueError("set GITHUB_TOKEN or GITHUB_TOKENS")
    client = GitHubClient(tokens=tokens)
    print(f"{len(repos)} repos in list: {skipped_done} already done, {skipped_failed} failed earlier "
          f"(use --retry-failed to retry), {len(todo)} to run with {workers} worker(s), "
          f"{len(tokens)} token(s)", flush=True)

    window = {"since": since, "until": until}

    def _work(name: str):
        owner, repo = name.split("/", 1)
        result = run_pipeline(owner, repo, since=since, until=until, client=client,
                              collect_prs=collect_prs, max_prs=max_prs, as_of=as_of,
                              collect_issues=collect_issues, max_issues=max_issues, max_commits=max_commits)
        export_jsonl(result, str(repo_dir(out_dir, name)))
        return result

    started = time.time()
    done = failed = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = {pool.submit(_work, name): name for name in todo}
    try:
        for fut in as_completed(futures):
            name = futures[fut]
            n = done + failed + 1
            try:
                result = fut.result()
                done += 1
                print(f"[{n}/{len(todo)}] {name}: {_one_line(result)}", flush=True)
            except Exception as exc:  # one bad repo never stops the corpus
                failed += 1
                log.debug("%s failed:\n%s", name, traceback.format_exc())
                _write_failure(out_dir, name, exc, window)
                print(f"[{n}/{len(todo)}] {name}: FAILED {exc.__class__.__name__}: {str(exc)[:200]}",
                      flush=True)
    except KeyboardInterrupt:
        print("\nStopping: waiting for running repos to finish; unstarted repos stay queued "
              "for the next run.", file=sys.stderr, flush=True)
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=True)
        raise
    pool.shutdown(wait=True)

    elapsed = time.time() - started
    print(f"Finished {done} ok, {failed} failed in {elapsed / 60:.1f} min"
          + f"; {client.requests_made} GitHub requests", flush=True)
    report = merge_corpus(out_dir)
    print(format_report(report), flush=True)
    print(f"Corpus files written to {out.resolve()}", flush=True)
    return report
