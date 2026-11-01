"""
Commit-level helpers shared by the API collectors: the code-size estimate
behind the 5,000-line rule (from file sizes in the tree listing), the
file-type classifier used by features.py, and edit velocity.

compute_edit_velocity() is a separate pass so it can be unit-tested against
a hand-built list of CommitRecord objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from . import config
from .schema import CommitRecord

_LOCK_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "gemfile.lock", "cargo.lock", "composer.lock", "pipfile.lock",
}


# "Lines of code": source and config files. Docs (.md/.rst/.txt) are left
# out, otherwise an awesome-list README counts as thousands of lines of code.
_CODE_EXTENSIONS = (
    config.FILE_TYPE_GROUPS["source"] | config.FILE_TYPE_GROUPS["config"]
    | {".sh", ".bash", ".ps1", ".css", ".scss", ".less", ".html", ".htm", ".vue",
       ".svelte", ".sql", ".m", ".mm", ".r", ".jl", ".lua", ".pl", ".pm", ".ex",
       ".exs", ".erl", ".hs", ".ml", ".clj", ".dart", ".groovy", ".gradle", ".vim",
       ".el", ".zig", ".nim", ".xml", ".proto", ".tf", ".cmake", ".mk"}
)


def _is_code_path(path: str) -> bool:
    base = path.rsplit("/", 1)[-1].lower()
    ext = "." + base.rsplit(".", 1)[-1] if "." in base else base
    return ext in _CODE_EXTENSIONS


def code_bytes(sizes: List[tuple]) -> int:
    """Bytes of the files count_loc would count, from (path, size) pairs
    -- available from a tree listing without reading any file."""
    return sum(size for path, size in sizes if size <= config.LOC_MAX_FILE_BYTES and _is_code_path(path))


def _classify_file(path: str) -> str:
    """Return one of 'source' / 'docs' / 'config' / 'test' / 'other'."""
    lower = path.lower()
    basename = lower.rsplit("/", 1)[-1]

    # test paths: path heuristic takes priority over extension
    if "/test" in lower or lower.startswith("test") or "/tests/" in lower or "_test." in basename or "test_" in basename:
        return "test"

    if basename in _LOCK_FILE_NAMES:
        return "config"

    ext = ""
    if "." in basename:
        ext = "." + basename.rsplit(".", 1)[-1]

    if basename == "dockerfile" or ext in config.FILE_TYPE_GROUPS["config"]:
        return "config"
    if ext in config.FILE_TYPE_GROUPS["docs"]:
        return "docs"
    if ext in config.FILE_TYPE_GROUPS["source"]:
        return "source"
    return "other"


def _is_mechanical_diff(commit: CommitRecord) -> bool:
    if not commit.file_paths:
        return False
    basenames = [p.lower().rsplit("/", 1)[-1] for p in commit.file_paths]
    if all(b in _LOCK_FILE_NAMES for b in basenames):
        return True
    # bulk reformatting heuristic: single file, enormous line count relative
    # to typical commits is handled by the percentile flag itself; here we
    # only catch the "all files are the same generated artifact" case.
    return False


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# an aware sentinel: comparing datetime.min (naive) with git's
# offset-carrying dates raises TypeError as soon as one date is missing
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


def compute_edit_velocity(commits: List[CommitRecord],
                          percentile: int = config.EDIT_VELOCITY_FLAG_PERCENTILE) -> None:
    """Annotate commits in place with seconds_since_author_prev_commit,
    edit_velocity, edit_velocity_flagged and edit_velocity_excluded_reason.

    Grouped per author e-mail (falls back to author name), sorted by
    author_date. Squash-merge commits and mechanical diffs (lockfiles)
    are excluded from the percentile ranking per the methodology's note,
    but still get an excluded_reason recorded rather than being dropped.
    Merge commits are excluded too: git reports no diff stats for them.
    """
    def _author_key(c: CommitRecord) -> str:
        return (c.author_email or c.author_name or "unknown").lower()

    by_author: dict[str, List[CommitRecord]] = {}
    for c in commits:
        by_author.setdefault(_author_key(c), []).append(c)

    velocities_eligible: List[float] = []

    for author, author_commits in by_author.items():
        author_commits.sort(key=lambda c: _parse_dt(c.author_date) or _EARLIEST)
        prev_dt: Optional[datetime] = None
        for c in author_commits:
            dt = _parse_dt(c.author_date)
            if prev_dt is not None and dt is not None:
                delta = (dt - prev_dt).total_seconds()
                if delta > 0:
                    c.seconds_since_author_prev_commit = delta
                    c.edit_velocity = (c.additions + c.deletions) / delta

                    excluded_reason = None
                    if c.merge_type == "squash":
                        excluded_reason = "squash_merge"
                    elif c.is_merge:
                        excluded_reason = "merge_commit"
                    elif _is_mechanical_diff(c):
                        excluded_reason = "mechanical_diff"
                    c.edit_velocity_excluded_reason = excluded_reason

                    if excluded_reason is None:
                        velocities_eligible.append(c.edit_velocity)
            if dt is not None:
                prev_dt = dt

    if len(velocities_eligible) < 2:
        return

    threshold = _percentile(velocities_eligible, percentile)
    for c in commits:
        c.edit_velocity_flagged = bool(
            c.edit_velocity is not None
            and c.edit_velocity_excluded_reason is None
            and c.edit_velocity >= threshold
        )


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return float("inf")
    data = sorted(values)
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)
