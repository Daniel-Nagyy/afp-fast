"""
Constants agreed in Shared_Corpus_Methodology.tex.

Keeping these in one module means a group decision change (e.g. adding
Cursor back in, changing the LOC threshold) is a one-line edit instead
of a hunt through the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Repo eligibility (Table: "Repo eligibility and time window parameters")
# ---------------------------------------------------------------------------

MIN_LOC = 5_000                   # lines of text in recognised source/doc/config files
MIN_COMMITS = 100
MIN_STARS = 10
ALLOW_FORKS = False
PUSH_RECENCY_MONTHS = 12          # "active/pushed in last 12 months"
DATA_COLLECTION_START = "2026-02-22"  # Robbes et al. cover up to 2026-02-21

SIZE_TIER_BOUNDARIES = {
    # (min_loc_inclusive, max_loc_exclusive) -> tier label
    "small": (5_000, 20_000),
    "medium": (20_000, 100_000),
    "large": (100_000, None),
}

# ---------------------------------------------------------------------------
# Agent identification (Table: "Agent identification parameters")
# ---------------------------------------------------------------------------

# Top five brands per Robbes et al. (2026) adoption figures. The full
# roster (bot logins, e-mails, trailers, body markers, branch prefixes,
# labels, file markers) lives in signatures.py; every other brand found
# there is still detected and reported, just outside the primary five.
from .signatures import PRIMARY_BRANDS as _PRIMARY_BRANDS

AGENT_BRANDS = list(_PRIMARY_BRANDS)

# Label-source priority, checked in this order (per methodology Notes
# item 2: trailer > bot account > PR marker). "message_marker" (an
# agent-written line such as "Generated with Claude Code" in the commit
# body) sits with the trailer as commit-text evidence; "email_domain"
# (an agent's exact author/committer address) is the git-side form of a
# bot account.
LABEL_SOURCE_PRIORITY = ["trailer", "message_marker", "bot_account", "email_domain", "pr_marker"]

LABEL_AGENT = "agent_labeled"
LABEL_UNLABELED = "unlabeled"   # never "human" -- see Positive/Unlabeled framing

# ---------------------------------------------------------------------------
# Commit-level features (Table: "Commit-level feature parameters")
# ---------------------------------------------------------------------------

CONVENTIONAL_COMMIT_PREFIXES = (
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build",
    "ci", "chore", "revert",
)

FILE_TYPE_GROUPS = {
    "source": {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h",
        ".hpp", ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".scala",
    },
    "docs": {".md", ".rst", ".txt", ".adoc"},
    "config": {
        ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".env",
        "dockerfile", ".lock",
    },
    "test": set(),  # determined by path heuristics (see features.py), not extension
}

EDIT_VELOCITY_FLAG_PERCENTILE = 95  # top 5% flagged as high-velocity

# ---------------------------------------------------------------------------
# PR-level features (Table: "Pull request-level feature parameters")
# ---------------------------------------------------------------------------

RUBBER_STAMP_MAX_COMMENTS = 0
RUBBER_STAMP_MAX_LATENCY_SECONDS = 15 * 60  # "fast approval" -- 15 minutes; group to confirm

# ---------------------------------------------------------------------------
# Collection limits (keep corpus-scale runs inside the API budget)
# ---------------------------------------------------------------------------

PR_PAGE_SIZE = 50             # PRs per GraphQL page (reviews nested)
MAX_PRS_PER_REPO = 300        # newest PRs collected per repo; 0 = no cap
LOC_MAX_FILE_BYTES = 2_000_000  # files larger than this are not line-counted
# api mode has no file contents: LOC is estimated as code bytes / this.
# Calibrated on the 185-repo corpus (Oct 2026): median 34.1 bytes per
# counted line (IQR 29.9-39.5); 32 gives the best agreement with the exact
# count on the 5,000-line rule, 172 of 177 repos (97%).
BYTES_PER_LOC = 32.0

# ---------------------------------------------------------------------------
# Stack / language tagging
# ---------------------------------------------------------------------------

LANGUAGE_TO_STACK_TAG = {
    "Python": "Python",
    "JavaScript": "JS/TS",
    "TypeScript": "JS/TS",
    "Java": "Java",
    "PHP": "PHP",
    "Rust": "Rust",
    "C#": "C#",
}
