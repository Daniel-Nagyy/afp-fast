"""
Shared data structures for the Agent Fingerprint Pipeline (AFP).

These mirror the parameters agreed in Shared_Corpus_Methodology.tex
(repo eligibility, agent identification, commit-level features,
PR-level features, stack/language tagging) so that every stage of
the pipeline writes and reads the same fields.

Nothing here performs I/O. Dataclasses are kept intentionally close
to "plain dicts with a schema" so they serialize to JSONL/CSV cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# --------------------------------------------------------------------------
# Repo-level
# --------------------------------------------------------------------------

@dataclass
class RepoMetadata:
    owner: str
    name: str
    full_name: str
    stars: int = 0
    is_fork: bool = False
    is_archived: bool = False
    pushed_at: Optional[str] = None          # ISO 8601
    created_at: Optional[str] = None         # ISO 8601
    primary_language: Optional[str] = None
    languages: Dict[str, int] = field(default_factory=dict)   # bytes per language
    total_loc: Optional[int] = None          # lines in source/config files at HEAD (estimated from file sizes)
    language_bytes: Optional[int] = None     # sum of the languages API byte counts
    code_bytes: Optional[int] = None         # bytes of the source/config files counted for LOC
    loc_estimated: bool = False              # total_loc estimated from code_bytes (api mode)
    commit_count: Optional[int] = None       # full default-branch history, not just the window
    commit_count_source: Optional[str] = None  # "graphql" / "git" / None
    contributor_count: Optional[int] = None
    is_solo: Optional[bool] = None           # contributor_count == 1
    default_branch: str = "main"
    branch_protection: Optional[bool] = None  # review required on the default branch; None = unknown
    branch_protection_source: Optional[str] = None  # "rulesets" / "classic" / None
    stack_tag: Optional[str] = None           # coarse tag, e.g. "Python", "JS/TS"
    size_tier: Optional[str] = None           # "small" / "medium" / "large"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EligibilityResult:
    full_name: str
    eligible: Optional[bool]   # None = passed every rule that could be checked, some unchecked
    reasons: List[str] = field(default_factory=list)  # failed-check explanations
    unchecked: List[str] = field(default_factory=list)  # rules that could not be checked (e.g. LOC in API mode)
    checked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Commit-level
# --------------------------------------------------------------------------

@dataclass
class LabelSignal:
    """Raw, retained-as-metadata label evidence for one commit.

    Per the methodology's Agent Identification table, these signals are
    kept as metadata and stripped from the model's feature set before
    training -- the classifier must not simply relearn the heuristic.
    """
    has_trailer: bool = False
    trailer_text: Optional[str] = None
    message_marker_match: bool = False
    message_marker_text: Optional[str] = None
    is_bot_account: bool = False
    bot_account_login: Optional[str] = None
    email_domain_match: bool = False
    email_domain: Optional[str] = None
    pr_marker_match: bool = False
    pr_marker_text: Optional[str] = None

    agent_brand: Optional[str] = None   # resolved brand, or "unknown"
    label_source: Optional[str] = None  # "trailer" / "bot_account" / "email_domain" / "pr_marker" / None
    label: str = "unlabeled"            # "agent_labeled" or "unlabeled" -- never "human"
    brands: List[str] = field(default_factory=list)  # every brand any signal named
    propagated_from_pr: Optional[int] = None         # PR number when labelled via its merge

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommitRecord:
    repo_full_name: str
    sha: str
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    committer_name: Optional[str] = None
    committer_email: Optional[str] = None
    author_date: Optional[str] = None     # ISO 8601
    commit_date: Optional[str] = None     # ISO 8601, i.e. the "event date"
    message: str = ""
    parents: List[str] = field(default_factory=list)
    is_merge: bool = False
    merge_type: Optional[str] = None      # "merge" / "squash" / "rebase" / None (regular commit)

    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    file_paths: List[str] = field(default_factory=list)

    seconds_since_author_prev_commit: Optional[float] = None
    edit_velocity: Optional[float] = None          # (additions+deletions) / seconds_since_prev
    edit_velocity_flagged: bool = False            # above the chosen percentile
    edit_velocity_excluded_reason: Optional[str] = None  # "squash_merge" / "mechanical_diff" / None

    label: LabelSignal = field(default_factory=LabelSignal)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# Pull-request-level
# --------------------------------------------------------------------------

@dataclass
class PullRequestRecord:
    repo_full_name: str
    number: int
    title: str = ""
    body: Optional[str] = None
    branch_name: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    author_login: Optional[str] = None
    created_at: Optional[str] = None
    merged_at: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    merged_via: Optional[str] = None       # "merge" / "squash" / "rebase" / "squash_or_rebase"
    commit_count: Optional[int] = None     # commits on the PR branch

    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    pr_size: int = 0                        # additions + deletions

    review_comment_count: int = 0
    review_depth: Optional[float] = None    # review_comment_count / pr_size
    first_review_at: Optional[str] = None   # first review by someone other than the author or a bot
    review_latency_seconds: Optional[float] = None
    first_approval_at: Optional[str] = None
    approval_latency_seconds: Optional[float] = None
    rubber_stamp_flag: bool = False         # zero comments + fast approval
    ai_reviewers: List[str] = field(default_factory=list)  # AI review bots that reviewed or commented
    mention_brands: List[str] = field(default_factory=list)          # agents @mentioned in body/comments
    agent_commenter_brands: List[str] = field(default_factory=list)  # agent bots that commented

    label: LabelSignal = field(default_factory=LabelSignal)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Feature vector (post explicit-signal stripping)
# --------------------------------------------------------------------------

@dataclass
class CommitFeatureVector:
    """Structural/statistical features only -- no label-source columns.

    This is what actually gets handed to the classifier. Keeping it
    separate from CommitRecord makes the "strip the signals before
    training" step an explicit, auditable transformation rather than
    an implicit convention someone can forget.
    """
    repo_full_name: str
    sha: str

    # message features
    message_length: int = 0
    message_line_count: int = 1
    has_conventional_prefix: bool = False
    template_likeness: float = 0.0

    # timing features
    seconds_since_prev_commit: Optional[float] = None
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None

    # diff features
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    add_delete_ratio: Optional[float] = None

    # file-type mix
    frac_source: float = 0.0
    frac_docs: float = 0.0
    frac_config: float = 0.0
    frac_test: float = 0.0

    # merge / edit velocity
    is_merge: bool = False
    merge_type: Optional[str] = None
    edit_velocity: Optional[float] = None
    edit_velocity_flagged: bool = False

    # training target -- PU-learning framing: 1 = agent_labeled, 0 = unlabeled
    # (never a ground-truth "0 = human"); kept here only for training/eval
    # convenience and must be dropped for genuine inference on new data.
    pu_label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Repo-level adoption roll-up
# --------------------------------------------------------------------------

@dataclass
class FileMarker:
    path: str
    marker: str
    brand: str
    first_added_at: Optional[str] = None   # earliest commit in the collected history adding it
    first_added_sha: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoSignal:
    """A repo-level adoption trace that is not a rules/guidance file: a CI
    workflow running an agent, MCP config, an agent dependency or dev
    container feature, a .gitignore entry hiding agent files, a setting
    that switches commit attribution off, or an agent sub-folder
    (sub-agents, commands, skills, prompts)."""
    kind: str            # workflow / mcp / dependency / devcontainer / gitignore / hidden_attribution / agent_dir
    brand: str
    path: str
    detail: str = ""
    strength: str = "medium"   # strong / medium / weak
    first_added_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IssueRecord:
    """An issue's agent involvement: who it was assigned to, which agents
    were @mentioned, and which agent bots commented."""
    repo_full_name: str
    number: int
    created_at: Optional[str] = None
    author_login: Optional[str] = None
    author_brand: Optional[str] = None
    assignee_brands: List[str] = field(default_factory=list)
    mention_brands: List[str] = field(default_factory=list)
    agent_commenter_brands: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoAdoption:
    """One row per repo: which agents left traces, through which source,
    and how much of the collected window they account for."""
    full_name: str
    commits_in_window: int = 0
    agent_commits: int = 0
    agent_commit_share: Optional[float] = None
    prs_collected: int = 0
    agent_prs: int = 0
    agent_prs_merged: int = 0     # merged agent PRs: adoption by the maintainers, not drive-by PRs
    agent_pr_share: Optional[float] = None
    brands: List[str] = field(default_factory=list)          # every brand seen anywhere
    primary_brands: List[str] = field(default_factory=list)  # restricted to the methodology's five
    authoring_brands: List[str] = field(default_factory=list)  # brands with commit/PR authorship evidence
    commit_brand_counts: Dict[str, int] = field(default_factory=dict)
    pr_brand_counts: Dict[str, int] = field(default_factory=dict)
    file_marker_brands: List[str] = field(default_factory=list)
    signal_brands: Dict[str, List[str]] = field(default_factory=dict)  # signal kind -> brands
    issue_brands: Dict[str, List[str]] = field(default_factory=dict)   # assigned / mentioned / commented -> brands
    label_source_counts: Dict[str, int] = field(default_factory=dict)
    first_agent_commit_at: Optional[str] = None
    adoption_date: Optional[str] = None          # earliest commit, PR, file or workflow evidence
    adoption_date_source: Optional[str] = None
    ai_reviewers: List[str] = field(default_factory=list)
    hidden_use: bool = False     # agent files ignored or attribution switched off

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
