"""
Structural/statistical feature extraction, per "Commit-level feature
parameters". Turns a CommitRecord into a CommitFeatureVector with every
explicit label-source signal stripped out -- this is the boundary that
keeps the PU-learning setup honest (schema.py's docstring on
CommitFeatureVector explains why).

Message features are computed on the message with every explicit agent
signal removed first (agent co-author trailers, "Generated with ..."
lines, task links, an "aider: " prefix). Otherwise message length and
line count would carry the label itself -- a Claude Code commit is two
lines longer purely because of its trailer -- and the classifier would
relearn the heuristic instead of a fingerprint.

compute_commit_features() assumes compute_edit_velocity() (collect.py)
has already annotated the commits it's given, since edit velocity needs
the whole per-author commit sequence and can't be computed one commit
at a time.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from . import config
from . import signatures as sig_db
from .collect import _classify_file  # reuse the same file-type classifier
from .heuristics import _trailer_brand
from .schema import CommitRecord, CommitFeatureVector

_CONVENTIONAL_PREFIX_RE = re.compile(
    r"^(" + "|".join(config.CONVENTIONAL_COMMIT_PREFIXES) + r")(\(.+?\))?!?:\s",
    re.IGNORECASE,
)

_TEMPLATE_MARKERS = [
    r"^merge branch",
    r"^merge pull request",
    r"^signed-off-by:",
    r"^this is an automated",
    r"^auto-generated",
    r"^bump .+ from .+ to .+",   # dependabot-style
]
_TEMPLATE_MARKER_RES = [re.compile(p, re.IGNORECASE) for p in _TEMPLATE_MARKERS]


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


_ROBOT_LINE_RE = re.compile(r"^\s*\U0001F916\s*generated (?:with|by)\b.*$", re.IGNORECASE)
# every line of the tool trailer families, including the ones that are not
# a label on their own (Replit writes Replit-Commit-* on human commits too)
_TOOL_TRAILER_LINE_RE = re.compile(
    r"^\s*(?:agent-logs-url|claude-session|amp-thread-id|replit-commit-[\w-]+|x-lovable-[\w-]+|made-with)\s*:",
    re.IGNORECASE)


def strip_label_signals(message: str) -> str:
    """The message with every explicit agent signal removed (see the
    module docstring). Human co-author trailers are kept."""
    if not message:
        return ""
    kept = []
    for i, line in enumerate(message.splitlines()):
        m = sig_db.TRAILER_RE.match(line)
        if m and _trailer_brand(m.group("name") or "", m.group("email") or ""):
            continue
        if _ROBOT_LINE_RE.match(line) or _TOOL_TRAILER_LINE_RE.match(line) \
                or any(p.search(line) for _, p in sig_db.BODY_MARKER_PATTERNS):
            continue
        if i == 0:
            for _, p in sig_db.SUBJECT_PREFIX_PATTERNS:
                line = p.sub("", line, count=1)
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _message_features(message: str):
    length = len(message)
    line_count = max(1, message.count("\n") + 1)
    first_line = message.splitlines()[0] if message else ""
    has_prefix = bool(_CONVENTIONAL_PREFIX_RE.match(first_line))

    marker_hits = sum(1 for r in _TEMPLATE_MARKER_RES if r.search(message))
    # crude 0..1 template-likeness: marker hit is a strong signal; very
    # short, generic first lines ("fix", "update", "wip") nudge it up too.
    generic_short = first_line.strip().lower() in {"fix", "update", "wip", "misc", "cleanup"}
    template_likeness = min(1.0, 0.6 * marker_hits + (0.3 if generic_short else 0.0))

    return length, line_count, has_prefix, template_likeness


def _file_type_mix(file_paths: List[str]):
    if not file_paths:
        return 0.0, 0.0, 0.0, 0.0
    counts = {"source": 0, "docs": 0, "config": 0, "test": 0, "other": 0}
    for p in file_paths:
        counts[_classify_file(p)] += 1
    n = len(file_paths)
    return (
        counts["source"] / n,
        counts["docs"] / n,
        counts["config"] / n,
        counts["test"] / n,
    )


def commit_to_feature_vector(commit: CommitRecord) -> CommitFeatureVector:
    length, line_count, has_prefix, template_likeness = _message_features(strip_label_signals(commit.message))
    frac_source, frac_docs, frac_config, frac_test = _file_type_mix(commit.file_paths)

    # author date: when (and in which timezone) the change was written.
    # The committer date of a squash or rebase merge is GitHub's merge time.
    dt = _parse_dt(commit.author_date or commit.commit_date)
    hour_of_day = dt.hour if dt else None
    day_of_week = dt.weekday() if dt else None

    add_delete_ratio = None
    if commit.deletions > 0:
        add_delete_ratio = commit.additions / commit.deletions
    elif commit.additions > 0:
        add_delete_ratio = float("inf")

    pu_label = 1 if commit.label.label == config.LABEL_AGENT else 0

    return CommitFeatureVector(
        repo_full_name=commit.repo_full_name,
        sha=commit.sha,
        message_length=length,
        message_line_count=line_count,
        has_conventional_prefix=has_prefix,
        template_likeness=template_likeness,
        seconds_since_prev_commit=commit.seconds_since_author_prev_commit,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        additions=commit.additions,
        deletions=commit.deletions,
        files_changed=commit.files_changed,
        add_delete_ratio=add_delete_ratio,
        frac_source=frac_source,
        frac_docs=frac_docs,
        frac_config=frac_config,
        frac_test=frac_test,
        is_merge=commit.is_merge,
        merge_type=commit.merge_type,
        edit_velocity=commit.edit_velocity,
        edit_velocity_flagged=commit.edit_velocity_flagged,
        pu_label=pu_label,
    )


def commits_to_feature_vectors(commits: List[CommitRecord]) -> List[CommitFeatureVector]:
    return [commit_to_feature_vector(c) for c in commits]
