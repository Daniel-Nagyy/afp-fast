"""
Command-line entry point for the Agent Fingerprint Pipeline.

Usage:
    python -m afp.cli run owner/repo --out ./out/owner_repo [--since 2026-02-22] [--until ...]
    python -m afp.cli batch corpus.csv --out ./out/corpus --since 2026-02-22 --workers 6
    python -m afp.cli report ./out/corpus
    python -m afp.cli smoke-test-api owner/repo
    python -m afp.cli train ./out/corpus/features.jsonl --out model.joblib
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from .pipeline import run_pipeline, export_jsonl

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _date(value: str) -> str:
    if not _DATE_RE.match(value):
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}")
    return value


def _cmd_run(args: argparse.Namespace) -> int:
    owner, repo = args.repo.split("/", 1)
    result = run_pipeline(
        owner, repo,
        since=args.since,
        until=args.until,
        collect_prs=not args.no_prs,
        max_prs=args.max_prs,
        as_of=args.as_of,
        collect_issues=not args.no_issues,
        max_issues=args.max_issues,
        max_commits=args.max_commits,
    )
    paths = export_jsonl(result, args.out)
    for note in result.notes:
        print(f"Note: {note}")
    print(f"Commits collected: {len(result.commits)}")
    print(f"PRs collected: {len(result.pull_requests)}")
    print(f"Feature vectors: {len(result.feature_vectors)}")
    if result.eligibility:
        verdict = "NOT ELIGIBLE (skipped)" if result.eligibility.eligible is False else "ELIGIBLE"
        print(f"Eligibility: {verdict} ({', '.join(result.eligibility.reasons) or 'no issues'})")
    ad = result.adoption
    if ad:
        print(f"Agent commits: {ad.agent_commits} of {ad.commits_in_window}; agent PRs: "
              f"{ad.agent_prs} of {ad.prs_collected}; brands: {', '.join(ad.brands) or 'none'}")
    print("Wrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from .batch import read_repo_list, run_batch
    repos = read_repo_list(args.repos)
    if not repos:
        print(f"No owner/repo names found in {args.repos}", file=sys.stderr)
        return 1
    try:
        run_batch(repos, args.out, workers=args.workers, since=args.since, until=args.until,
                  collect_prs=not args.no_prs, max_prs=args.max_prs, retry_failed=args.retry_failed,
                  limit=args.limit, as_of=args.as_of, collect_issues=not args.no_issues,
                  max_issues=args.max_issues, max_commits=args.max_commits)
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .adoption import format_report, merge_corpus
    print(format_report(merge_corpus(args.out_dir)))
    return 0


def _cmd_smoke_test_api(args: argparse.Namespace) -> int:
    """A tiny live check of the API layer against the real GitHub API."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set. Export a GitHub personal access token "
              "(no special scopes needed for public repos) and re-run.", file=sys.stderr)
        return 1

    from .eligibility import build_repo_metadata
    from .github_client import GitHubClient
    from .pr_collect import collect_pull_requests
    owner, repo = args.repo.split("/", 1)
    client = GitHubClient(token=token)

    print(f"GraphQL repository {owner}/{repo} + contributors + review rules ...")
    meta = build_repo_metadata(client, owner, repo)
    print(f"  full_name={meta.full_name} stars={meta.stars} fork={meta.is_fork} "
          f"archived={meta.is_archived} pushed_at={meta.pushed_at} language={meta.primary_language} "
          f"commits={meta.commit_count} contributors={meta.contributor_count} "
          f"review_required={meta.branch_protection} ({meta.branch_protection_source})")

    print("GraphQL pull requests (one page) ...")
    prs = collect_pull_requests(client, owner, repo, max_prs=20)
    agent = [p for p in prs if p.label.label == "agent_labeled"]
    print(f"  fetched {len(prs)} PRs, {len(agent)} agent-labelled")
    for p in agent[:5]:
        print(f"    #{p.number} {p.label.agent_brand} via {p.label.label_source}: {p.label.pr_marker_text or p.label.bot_account_login}")
    print(f"Smoke test passed: {client.requests_made} requests, live responses parsed without error.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .classifier import train_from_jsonl
    train_from_jsonl(args.features_jsonl, args.out, model_name=args.model)
    print(f"Model written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afp", description="Agent Fingerprint Pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="log retries and rate-limit waits")
    sub = parser.add_subparsers(dest="command", required=True)

    def window_args(p):
        p.add_argument("--since", type=_date, default=None,
                       help="first day of the window, YYYY-MM-DD (use the group's data-collection start date)")
        p.add_argument("--until", type=_date, default=None, help="last day of the window, YYYY-MM-DD")
        p.add_argument("--no-prs", action="store_true", help="skip PR-level collection even if a token is set")
        p.add_argument("--max-prs", type=int, default=None,
                       help="newest PRs collected per repo (default 300; 0 = all in the window)")
        p.add_argument("--no-issues", action="store_true", help="skip issue collection (assignments, @mentions)")
        p.add_argument("--max-issues", type=int, default=200, help="newest issues scanned per repo (0 = all in the window)")
        p.add_argument("--max-commits", type=int, default=2000, help="newest commits per repo (0 = all)")
        p.add_argument("--as-of", type=_date, default=None,
                       help="date the 12-month push-recency rule is measured from (default today)")

    p_run = sub.add_parser("run", help="Run the full pipeline on one repo")
    p_run.add_argument("repo", help="owner/repo, e.g. octocat/Hello-World")
    p_run.add_argument("--out", required=True, help="output directory for JSONL/summary files")
    window_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_batch = sub.add_parser("batch", help="Run the pipeline over a corpus (resumable)")
    p_batch.add_argument("repos", help="corpus CSV with a full_name column (e.g. a gitcorpus export) "
                                       "or a text file with one owner/repo per line")
    p_batch.add_argument("--out", required=True, help="output directory for the corpus")
    p_batch.add_argument("--workers", type=int, default=None,
                         help="repos processed in parallel (default 24)")
    p_batch.add_argument("--retry-failed", action="store_true", help="re-run repos that failed before")
    p_batch.add_argument("--limit", type=int, default=None, help="only run the first N pending repos (pilot)")
    window_args(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    p_report = sub.add_parser("report", help="Re-merge a batch output directory and print the summary")
    p_report.add_argument("out_dir")
    p_report.set_defaults(func=_cmd_report)

    p_smoke = sub.add_parser("smoke-test-api", help="Live-check the API layer (needs GITHUB_TOKEN)")
    p_smoke.add_argument("repo", help="owner/repo")
    p_smoke.set_defaults(func=_cmd_smoke_test_api)

    p_train = sub.add_parser("train", help="Train the baseline classifier from a features.jsonl file")
    p_train.add_argument("features_jsonl", help="path to a features.jsonl produced by `run` or `batch`")
    p_train.add_argument("--out", default="model.joblib", help="where to write the trained model")
    p_train.add_argument("--model", choices=("logreg", "gboost"), default="logreg")
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
