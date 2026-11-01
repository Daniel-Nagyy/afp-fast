"""
Adoption roll-ups: per repo (which agents left traces, through which
channel, what share of the window they account for, when adoption
started) and per corpus (merging every repo's output into combined
files and one table per brand and channel).

Channels (see docs/agent-adoption-guide.pdf): commit, pr, file,
workflow, mcp, agent_dir, devcontainer, gitignore, hidden_attribution,
dependency, issue (assigned / agent bot commented), mention.
A brand counts as adopted by a repo when any channel except the weak
ones (dependency, template-like .gitignore, unanswered @mention) shows
it. Weak channels are still reported, in their own columns.

Everything here reads records the pipeline already produced; nothing
calls git or the API.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import config
from .schema import CommitRecord, FileMarker, IssueRecord, PullRequestRecord, RepoAdoption, RepoSignal
from .signatures import BRAND_KIND, BRAND_NAMES, PRIMARY_BRANDS

CHANNELS = ("commit", "pr", "file", "workflow", "mcp", "agent_dir", "devcontainer", "gitignore",
            "hidden_attribution", "dependency", "issue", "mention")


def _add(target: List[str], brand: Optional[str]) -> None:
    if brand and brand not in target:
        target.append(brand)


def summarize_repo(full_name: str, commits: List[CommitRecord], prs: List[PullRequestRecord],
                   markers: List[FileMarker], signals: Optional[List[RepoSignal]] = None,
                   issues: Optional[List[IssueRecord]] = None) -> RepoAdoption:
    signals, issues = signals or [], issues or []
    agent_commits = [c for c in commits if c.label.label == config.LABEL_AGENT]
    agent_prs = [p for p in prs if p.label.label == config.LABEL_AGENT]
    commit_brands = Counter(c.label.agent_brand for c in agent_commits)
    pr_brands = Counter(p.label.agent_brand for p in agent_prs)
    sources = Counter(c.label.label_source for c in agent_commits)
    sources.update(f"pr:{p.label.label_source}" for p in agent_prs)
    marker_brands = sorted({m.brand for m in markers})

    by_channel: Dict[str, List[str]] = {ch: [] for ch in CHANNELS}
    for brand in commit_brands:
        _add(by_channel["commit"], brand)
    for c in agent_commits:  # secondary brands a commit also named
        for b in c.label.brands:
            _add(by_channel["commit"], b)
    for brand in pr_brands:
        _add(by_channel["pr"], brand)
    for brand in marker_brands:
        _add(by_channel["file"], brand)
    for s in signals:
        _add(by_channel[s.kind], s.brand)
    weak_gitignore = {s.brand for s in signals if s.kind == "gitignore" and s.strength == "weak"}
    issue_brands = {"assigned": [], "agent_commented": [], "mentioned": []}
    for i in issues:
        for b in i.assignee_brands + ([i.author_brand] if i.author_brand else []):
            _add(issue_brands["assigned"], b)
        for b in i.agent_commenter_brands:
            _add(issue_brands["agent_commented"], b)
        for b in i.mention_brands:
            _add(issue_brands["mentioned"], b)
    for p in prs:
        for b in p.agent_commenter_brands:
            _add(issue_brands["agent_commented"], b)
        for b in p.mention_brands:
            _add(issue_brands["mentioned"], b)
    for b in issue_brands["assigned"] + issue_brands["agent_commented"]:
        _add(by_channel["issue"], b)
    confirmed = set(by_channel["issue"]) | set(by_channel["commit"]) | set(by_channel["pr"])
    for b in issue_brands["mentioned"]:
        _add(by_channel["mention"], b)

    brands: List[str] = []
    for ch in ("commit", "pr", "file", "workflow", "mcp", "agent_dir", "devcontainer",
               "hidden_attribution", "issue"):
        for b in by_channel[ch]:
            _add(brands, b)
    for b in by_channel["gitignore"]:
        if b not in weak_gitignore:
            _add(brands, b)
    for b in by_channel["mention"]:
        if b in confirmed:
            _add(brands, b)

    authoring = []
    for b in by_channel["commit"] + by_channel["pr"]:
        if BRAND_KIND.get(b) not in ("review", "tooling"):
            _add(authoring, b)

    # adoption date: earliest dated evidence, and which channel gave it
    dated = [(c.author_date, "commit") for c in agent_commits if c.author_date]
    dated += [(p.created_at, "pr") for p in agent_prs if p.created_at]
    dated += [(m.first_added_at, "file") for m in markers if m.first_added_at]
    dated += [(s.first_added_at, s.kind) for s in signals if s.first_added_at]
    dated.sort(key=lambda d: d[0])
    commit_dates = sorted(c.author_date for c in agent_commits if c.author_date)

    reviewers: List[str] = []
    for p in prs:
        for r in p.ai_reviewers:
            _add(reviewers, r)

    return RepoAdoption(
        full_name=full_name,
        commits_in_window=len(commits),
        agent_commits=len(agent_commits),
        agent_commit_share=(len(agent_commits) / len(commits)) if commits else None,
        prs_collected=len(prs),
        agent_prs=len(agent_prs),
        agent_prs_merged=sum(1 for p in agent_prs if p.merged_at),
        agent_pr_share=(len(agent_prs) / len(prs)) if prs else None,
        brands=brands,
        primary_brands=[b for b in brands if b in PRIMARY_BRANDS],
        authoring_brands=authoring,
        commit_brand_counts=dict(commit_brands),
        pr_brand_counts=dict(pr_brands),
        file_marker_brands=marker_brands,
        signal_brands={ch: by_channel[ch] for ch in ("workflow", "mcp", "agent_dir", "devcontainer",
                                                     "gitignore", "hidden_attribution", "dependency")
                       if by_channel[ch]},
        issue_brands={k: v for k, v in issue_brands.items() if v},
        label_source_counts=dict(sources),
        first_agent_commit_at=commit_dates[0] if commit_dates else None,
        adoption_date=dated[0][0] if dated else None,
        adoption_date_source=dated[0][1] if dated else None,
        ai_reviewers=reviewers,
        hidden_use=any(s.kind == "hidden_attribution" or (s.kind == "gitignore" and s.strength != "weak")
                       for s in signals),
    )


# ---------------------------------------------------------------------------
# Corpus-level merge
# ---------------------------------------------------------------------------

REPO_COLUMNS = [
    "full_name", "status", "error", "mode", "eligible", "eligibility_reasons", "eligibility_unchecked",
    "stars", "primary_language", "stack_tag", "total_loc", "size_tier", "commit_count",
    "contributor_count", "is_solo", "branch_protection", "pushed_at",
    "commits_in_window", "agent_commits", "agent_commit_share",
    "prs_collected", "agent_prs", "agent_prs_merged", "agent_pr_share",
    "any_agent", "brands", "primary_brands", "authoring_brands", "file_marker_brands",
    "workflow_brands", "mcp_brands", "agent_dir_brands", "devcontainer_brands", "gitignore_brands",
    "hidden_attribution_brands", "dependency_brands", "issue_assigned_brands",
    "issue_agent_commented_brands", "mentioned_brands", "hidden_use",
    "first_agent_commit_at", "adoption_date", "adoption_date_source", "ai_reviewers", "label_sources",
]


def _flat_repo_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    meta = summary.get("repo_metadata") or {}
    elig = summary.get("eligibility") or {}
    ad = summary.get("adoption") or {}
    sig = ad.get("signal_brands") or {}
    iss = ad.get("issue_brands") or {}
    j = lambda xs: ";".join(xs or [])
    return {
        "full_name": summary.get("full_name"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "mode": summary.get("mode"),
        "eligible": elig.get("eligible"),
        "eligibility_reasons": "; ".join(elig.get("reasons") or []),
        "eligibility_unchecked": "; ".join(elig.get("unchecked") or []),
        "stars": meta.get("stars"),
        "primary_language": meta.get("primary_language"),
        "stack_tag": meta.get("stack_tag"),
        "total_loc": meta.get("total_loc"),
        "size_tier": meta.get("size_tier"),
        "commit_count": meta.get("commit_count"),
        "contributor_count": meta.get("contributor_count"),
        "is_solo": meta.get("is_solo"),
        "branch_protection": meta.get("branch_protection"),
        "pushed_at": meta.get("pushed_at"),
        "commits_in_window": ad.get("commits_in_window"),
        "agent_commits": ad.get("agent_commits"),
        "agent_commit_share": ad.get("agent_commit_share"),
        "prs_collected": ad.get("prs_collected"),
        "agent_prs": ad.get("agent_prs"),
        "agent_prs_merged": ad.get("agent_prs_merged"),
        "agent_pr_share": ad.get("agent_pr_share"),
        "any_agent": bool(ad.get("brands")) if ad else None,
        "brands": j(ad.get("brands")),
        "primary_brands": j(ad.get("primary_brands")),
        "authoring_brands": j(ad.get("authoring_brands")),
        "file_marker_brands": j(ad.get("file_marker_brands")),
        "workflow_brands": j(sig.get("workflow")),
        "mcp_brands": j(sig.get("mcp")),
        "agent_dir_brands": j(sig.get("agent_dir")),
        "devcontainer_brands": j(sig.get("devcontainer")),
        "gitignore_brands": j(sig.get("gitignore")),
        "hidden_attribution_brands": j(sig.get("hidden_attribution")),
        "dependency_brands": j(sig.get("dependency")),
        "issue_assigned_brands": j(iss.get("assigned")),
        "issue_agent_commented_brands": j(iss.get("agent_commented")),
        "mentioned_brands": j(iss.get("mentioned")),
        "hidden_use": ad.get("hidden_use"),
        "first_agent_commit_at": ad.get("first_agent_commit_at"),
        "adoption_date": ad.get("adoption_date"),
        "adoption_date_source": ad.get("adoption_date_source"),
        "ai_reviewers": j(ad.get("ai_reviewers")),
        "label_sources": ";".join(f"{k}={v}" for k, v in sorted((ad.get("label_source_counts") or {}).items())),
    }


def _concat_jsonl(sources: Iterable[Path], dest: Path) -> int:
    n = 0
    with open(dest, "w", encoding="utf-8", newline="\n") as out:
        for src in sources:
            if not src.exists():
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def _channel_presence(ad: Dict[str, Any], brand: str) -> Dict[str, bool]:
    sig = ad.get("signal_brands") or {}
    iss = ad.get("issue_brands") or {}
    return {
        "commit": brand in (ad.get("commit_brand_counts") or {}),
        "pr": brand in (ad.get("pr_brand_counts") or {}),
        "file": brand in (ad.get("file_marker_brands") or []),
        "workflow": brand in (sig.get("workflow") or []),
        "mcp": brand in (sig.get("mcp") or []),
        "agent_dir": brand in (sig.get("agent_dir") or []),
        "issue": brand in (iss.get("assigned") or []) or brand in (iss.get("agent_commented") or []),
        "hidden": brand in (sig.get("hidden_attribution") or []) or brand in (sig.get("gitignore") or []),
    }


def merge_corpus(out_dir: str) -> Dict[str, Any]:
    """Combine every repos/<owner__repo>/ output under `out_dir` into
    corpus-wide files and return the headline numbers:

      repos.csv              one row per repo: metadata, eligibility, adoption by channel
      adoption_by_brand.csv  repos per brand, split by channel, all repos and eligible only
      file_markers.csv       every agent config file found, with first-added date
      signals.csv            every repo signal (workflows, MCP, hidden use, ...)
      commits.jsonl, prs.jsonl, issues.jsonl, features.jsonl   all repos concatenated
    """
    out = Path(out_dir)
    repo_dirs = sorted(p for p in (out / "repos").glob("*") if p.is_dir())
    summaries = []
    for d in repo_dirs:
        s_path = d / "summary.json"
        if s_path.exists():
            with open(s_path, encoding="utf-8") as f:
                summaries.append(json.load(f))
    ok = [s for s in summaries if s.get("status") == "ok"]

    with open(out / "repos.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPO_COLUMNS)
        w.writeheader()
        for s in summaries:
            w.writerow(_flat_repo_row(s))

    channel_cols = ("commit", "pr", "file", "workflow", "mcp", "agent_dir", "issue", "hidden")

    def _brand_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        table: Dict[str, Dict[str, int]] = {}
        for s in rows:
            ad = s.get("adoption") or {}
            for brand in ad.get("brands") or []:
                t = table.setdefault(brand, {"repos": 0, "agent_commits": 0, "agent_prs": 0,
                                             **{f"via_{c}": 0 for c in channel_cols}})
                t["repos"] += 1
                t["agent_commits"] += (ad.get("commit_brand_counts") or {}).get(brand, 0)
                t["agent_prs"] += (ad.get("pr_brand_counts") or {}).get(brand, 0)
                for c, present in _channel_presence(ad, brand).items():
                    t[f"via_{c}"] += int(present)
        return table

    eligible = [s for s in ok if (s.get("eligibility") or {}).get("eligible")]
    all_table, elig_table = _brand_table(ok), _brand_table(eligible)
    with open(out / "adoption_by_brand.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["brand", "brand_name", "kind", "primary", "repos", "repo_share", "agent_commits", "agent_prs"]
                   + [f"repos_via_{c}" for c in channel_cols] + ["eligible_repos", "eligible_repo_share"])
        for brand in sorted(all_table, key=lambda b: (-all_table[b]["repos"], b)):
            t, e = all_table[brand], elig_table.get(brand, {"repos": 0})
            w.writerow([brand, BRAND_NAMES.get(brand, brand), BRAND_KIND.get(brand, ""), brand in PRIMARY_BRANDS,
                        t["repos"], round(t["repos"] / len(ok), 4) if ok else None, t["agent_commits"], t["agent_prs"]]
                       + [t[f"via_{c}"] for c in channel_cols]
                       + [e["repos"], round(e["repos"] / len(eligible), 4) if eligible else None])

    def _rows_from(name: str):
        for d in repo_dirs:
            p = d / name
            if p.exists():
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            yield d.name.replace("__", "/", 1), json.loads(line)

    with open(out / "file_markers.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["full_name", "brand", "marker", "path", "first_added_at", "first_added_sha"])
        for name, m in _rows_from("file_markers.jsonl"):
            w.writerow([name, m["brand"], m["marker"], m["path"], m.get("first_added_at"), m.get("first_added_sha")])

    with open(out / "signals.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["full_name", "kind", "brand", "path", "detail", "strength", "first_added_at"])
        for name, s in _rows_from("signals.jsonl"):
            w.writerow([name, s["kind"], s["brand"], s["path"], s.get("detail"), s.get("strength"),
                        s.get("first_added_at")])

    counts = {}
    for name in ("commits.jsonl", "prs.jsonl", "issues.jsonl", "features.jsonl"):
        counts[name] = _concat_jsonl((d / name for d in repo_dirs), out / name)

    any_agent = [s for s in ok if (s.get("adoption") or {}).get("brands")]
    report = {
        "repos_total": len(summaries),
        "repos_ok": len(ok),
        "repos_failed": sum(1 for s in summaries if s.get("status") == "failed"),
        "repos_skipped_ineligible": sum(1 for s in summaries if s.get("status") == "ineligible"),
        "repos_eligible": len(eligible),
        "repos_eligible_unchecked": sum(1 for s in ok if (s.get("eligibility") or {}).get("eligible") is None
                                        and s.get("eligibility")),
        "repos_with_any_agent": len(any_agent),
        "repos_with_primary_agent": sum(1 for s in ok if (s.get("adoption") or {}).get("primary_brands")),
        "repos_with_authoring_evidence": sum(1 for s in ok if (s.get("adoption") or {}).get("authoring_brands")),
        "repos_with_hidden_use": sum(1 for s in ok if (s.get("adoption") or {}).get("hidden_use")),
        "eligible_with_any_agent": sum(1 for s in eligible if (s.get("adoption") or {}).get("brands")),
        "brands": {b: t["repos"] for b, t in sorted(all_table.items(), key=lambda kv: -kv[1]["repos"])},
        "rows": counts,
    }
    with open(out / "corpus_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def format_report(report: Dict[str, Any]) -> str:
    ok = report["repos_ok"] or 1
    lines = [
        f"Repos processed: {report['repos_ok']} ok, {report['repos_failed']} failed"
        + (f", {report['repos_skipped_ineligible']} skipped as ineligible"
           if report.get("repos_skipped_ineligible") else "") + " "
        f"({report['repos_eligible']} meet every eligibility rule"
        + (f", {report['repos_eligible_unchecked']} pass the rules that could be checked"
           if report.get("repos_eligible_unchecked") else "") + ")",
        f"Repos with any agent trace: {report['repos_with_any_agent']} "
        f"({100 * report['repos_with_any_agent'] / ok:.1f}%); "
        f"with agent-written commits or PRs: {report.get('repos_with_authoring_evidence', 0)}; "
        f"hidden use: {report.get('repos_with_hidden_use', 0)}",
    ]
    for brand, n in report["brands"].items():
        lines.append(f"  {BRAND_NAMES.get(brand, brand):20} {n:5} repos")
    rows = report["rows"]
    lines.append(f"Rows: {rows.get('commits.jsonl', 0)} commits, {rows.get('prs.jsonl', 0)} PRs, "
                 f"{rows.get('issues.jsonl', 0)} agent issues, {rows.get('features.jsonl', 0)} feature vectors")
    return "\n".join(lines)
