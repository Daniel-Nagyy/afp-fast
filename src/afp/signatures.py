"""
Agent signatures: every explicit trace a coding agent is known to leave.

The roster and channel split follow Robbes et al. (2026)'s validated
heuristic set (file markers, commit trailers, branch names, PR labels),
extended with the bot accounts, e-mail addresses, tool trailers and
body markers the agents write themselves, each confirmed against live
GitHub data (October 2026; see docs/agent-adoption-guide.pdf). Every
pattern is *anchored* -- a brand name appearing anywhere in free text
is never a signal. That free-text matching ("move the cursor", a
co-author called Claude) was the main false-positive source when this
roster was first run on a real corpus.

Brands are snake_case ids. PRIMARY_BRANDS are the five the methodology
analyses; everything else is still detected and reported, so a repo
adopting e.g. Devin or Lovable is not silently counted as "no agent".
BRAND_KIND groups brands (cloud agent, IDE/terminal agent, app builder,
review bot, tooling). "unknown" is the methodology's catch-all for
generic agent traces (AGENTS.md, an "ai-generated" label) that do not
name a brand.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

PRIMARY_BRANDS = ("claude_code", "copilot", "codex", "gemini", "cursor")

# The five agents seen most in three 500-repo samples (Feb 2025 - Sep
# 2026): Claude Code (146 repos), Copilot (60), Codex (17), Cursor (17)
# and Jules (111 agent commits/PRs; Gemini CLI barely appeared). This
# tool detects only these (set at the bottom of this module).
TOP_BRANDS = ("claude_code", "copilot", "codex", "cursor", "jules")

# brand -> (display name, kind). Kinds: primary, cloud, ide, app_builder,
# review (AI code review / autofix bots), tooling (planning/spec tools
# used alongside agents), generic.
BRANDS: Dict[str, Tuple[str, str]] = {
    "claude_code": ("Claude Code", "primary"),
    "copilot": ("Copilot", "primary"),
    "codex": ("Codex", "primary"),
    "gemini": ("Gemini", "primary"),
    "cursor": ("Cursor", "primary"),
    # cloud / autonomous agents
    "jules": ("Jules", "cloud"),
    "antigravity": ("Google Antigravity", "cloud"),
    "devin": ("Devin", "cloud"),
    "openhands": ("OpenHands", "cloud"),
    "factory": ("Factory Droid", "cloud"),
    "amazon_q": ("Amazon Q Developer", "cloud"),
    "kiro": ("Kiro", "cloud"),
    "codegen": ("Codegen", "cloud"),
    "sweep": ("Sweep", "cloud"),
    "charlie": ("Charlie", "cloud"),
    "grok": ("Grok", "cloud"),
    "sentry_seer": ("Sentry Seer", "cloud"),
    "tembo": ("Tembo", "cloud"),
    "gru": ("Gru", "cloud"),
    "open_swe": ("Open SWE", "cloud"),
    "abacus": ("Abacus.AI", "cloud"),
    "amplifier": ("Microsoft Amplifier", "cloud"),
    # terminal and IDE agents
    "aider": ("Aider", "ide"),
    "amp": ("Amp", "ide"),
    "opencode": ("OpenCode", "ide"),
    "crush": ("Crush", "ide"),
    "qwen_coder": ("Qwen Code", "ide"),
    "warp": ("Warp", "ide"),
    "goose": ("Goose", "ide"),
    "junie": ("Junie", "ide"),
    "cline": ("Cline", "ide"),
    "roo_code": ("Roo Code", "ide"),
    "kilo_code": ("Kilo Code", "ide"),
    "windsurf": ("Windsurf", "ide"),
    "continue": ("Continue", "ide"),
    "augment": ("Augment Code", "ide"),
    "trae": ("TRAE", "ide"),
    "letta": ("Letta Code", "ide"),
    "mistral_vibe": ("Mistral Vibe", "ide"),
    "ona": ("Ona", "ide"),
    "pi": ("Pi", "ide"),
    "kimi": ("Kimi Code", "ide"),
    "sketch": ("Sketch", "ide"),
    "codebuddy": ("CodeBuddy", "ide"),
    "blackbox": ("Blackbox AI", "ide"),
    "tabnine": ("Tabnine", "ide"),
    "zencoder": ("Zencoder", "ide"),
    "chatgpt": ("ChatGPT", "ide"),
    "phoenix": ("Phoenix (Fly.io)", "ide"),
    "lingma": ("Alibaba Lingma", "ide"),
    "comate": ("Baidu Comate", "ide"),
    "brokk": ("Brokk", "ide"),
    "plandex": ("Plandex", "ide"),
    "rovo_dev": ("Atlassian Rovo Dev", "ide"),
    "verdent": ("Verdent", "ide"),
    "serena": ("Serena", "ide"),
    # app builders
    "lovable": ("Lovable", "app_builder"),
    "replit": ("Replit Agent", "app_builder"),
    "v0": ("v0", "app_builder"),
    "stitch": ("Google Stitch", "app_builder"),
    "bolt": ("Bolt.new", "app_builder"),
    # review / autofix bots (a commit they co-author applied an AI fix)
    "coderabbit": ("CodeRabbit", "review"),
    "qodo": ("Qodo", "review"),
    "ellipsis": ("Ellipsis", "review"),
    "sourcery": ("Sourcery", "review"),
    "deepsource": ("DeepSource Autofix", "review"),
    "greptile": ("Greptile", "review"),
    # tooling used with agents
    "speckit": ("Spec Kit", "tooling"),
    "taskmaster": ("Taskmaster", "tooling"),
    "tessl": ("Tessl", "tooling"),
    "rulesync": ("Rulesync", "tooling"),
    "unknown": ("Unknown agent", "generic"),
}

BRAND_NAMES: Dict[str, str] = {b: name for b, (name, _kind) in BRANDS.items()}
BRAND_KIND: Dict[str, str] = {b: kind for b, (_name, kind) in BRANDS.items()}

# ---------------------------------------------------------------------------
# Identities: exact e-mail addresses and GitHub logins
# ---------------------------------------------------------------------------

# Exact, lower-cased addresses. Whole domains are deliberately NOT mapped:
# openai.com / anthropic.com / google.com staff commit to open source as
# humans.
AGENT_EMAILS: Dict[str, str] = {
    "noreply@anthropic.com": "claude_code",
    "claude@anthropic.com": "claude_code",
    "assistant@anthropic.com": "claude_code",
    "cursoragent@cursor.com": "cursor",
    "codex@openai.com": "codex",
    "noreply@openai.com": "codex",
    "chatgpt@openai.com": "chatgpt",
    "chatgpt@chatgpt.com": "chatgpt",
    "aider@aider.chat": "aider",
    "noreply@aider.chat": "aider",
    "openhands@all-hands.dev": "openhands",
    "cascade@windsurf.ai": "windsurf",
    "noreply@opencode.ai": "opencode",
    "amp@ampcode.com": "amp",
    "kiro@aws.amazon.com": "kiro",
    "kiro-agent@amazon.com": "kiro",
    "kiro-noreply@amazon.com": "kiro",
    "noreply@warp.dev": "warp",
    "agent@warp.dev": "warp",
    "roomote@roocode.com": "roo_code",
    "crush@charm.land": "crush",
    "qwen-coder@alibabacloud.com": "qwen_coder",
    "vibe@mistral.ai": "mistral_vibe",
    "noreply@letta.com": "letta",
    "no-reply@ona.com": "ona",
    "junie@jetbrains.com": "junie",
    "noreply@continue.dev": "continue",
    "noreply@augmentcode.com": "augment",
    "traecli@bytedance.com": "trae",
    "noreply@cline.bot": "cline",
    "hello@sketch.dev": "sketch",
    "noreply@pi.dev": "pi",
    "charlie@charlielabs.ai": "charlie",
    "agent@replit.com": "replit",
    "it+v0agent@vercel.com": "v0",
    "noreply@tabnine.com": "tabnine",
    "noreply@zencoder.ai": "zencoder",
    "dev@fly.io": "phoenix",
}

# Addresses that identify the agent only together with its display name
# (the address alone is shared with other Google / Block tooling).
NAME_EMAIL_PAIRS: List[Tuple[Pattern[str], str, str]] = [
    (re.compile(r"^antigravity\b", re.I), "noreply@google.com", "antigravity"),
    (re.compile(r"^goose$", re.I), "opensource@block.xyz", "goose"),
]

# Tools that reuse another agent's address: the display name decides
# (e.g. Kimi Code writes `Kimi Code <noreply@anthropic.com>`).
TOOL_NAME_OVERRIDES: Dict[str, str] = {
    "kimi code": "kimi",
    "zencoder": "zencoder",
}

# Lower-cased GitHub logins of accounts that author code for an agent.
# Plain names ("claude", "cursor", "jules") are only trusted with the
# [bot] suffix, since humans hold those logins; "copilot" is GitHub's own
# reserved account.
AGENT_LOGINS: Dict[str, str] = {
    "copilot": "copilot",
    "copilot[bot]": "copilot",
    "copilot-swe-agent": "copilot",
    "copilot-swe-agent[bot]": "copilot",
    "claude[bot]": "claude_code",
    "chatgpt-codex-connector[bot]": "codex",
    "openai-codex[bot]": "codex",
    "codex[bot]": "codex",
    "cursor[bot]": "cursor",
    "gemini-code-assist[bot]": "gemini",
    "gemini-cli[bot]": "gemini",
    "google-labs-jules[bot]": "jules",
    "antigravity-ai": "antigravity",
    "devin-ai-integration[bot]": "devin",
    "openhands-agent": "openhands",
    "sweep-ai[bot]": "sweep",
    "jetbrains-junie[bot]": "junie",
    "amazon-q-developer[bot]": "amazon_q",
    "factory-droid[bot]": "factory",
    "windsurf-bot[bot]": "windsurf",
    "roomote-agent": "roo_code",
    "roomote": "roo_code",
    "roomote-roomote[bot]": "roo_code",
    "cline-cloud[bot]": "cline",
    "kiro-agent[bot]": "kiro",
    "kiro-agent": "kiro",
    "codegen-sh[bot]": "codegen",
    "charliecreates[bot]": "charlie",
    "grok-by-xai[bot]": "grok",
    "seer-by-sentry[bot]": "sentry_seer",
    "sentry-autofix[bot]": "sentry_seer",
    "tembo[bot]": "tembo",
    "tembo-io[bot]": "tembo",
    "gru-agent": "gru",
    "open-swe": "open_swe",
    "abacus-ai-automation[bot]": "abacus",
    "microsoft-amplifier": "amplifier",
    "gpt-engineer-app[bot]": "lovable",
    "lovable-dev[bot]": "lovable",
    "v0[bot]": "v0",
    "google-labs-stitch[bot]": "stitch",
    "tessl-helper[bot]": "tessl",
    "codebuddy": "codebuddy",
    "blackboxai": "blackbox",
    # review / autofix bots: a commit they co-author applied an AI fix
    "coderabbitai[bot]": "coderabbit",
    "qodo-code-review[bot]": "qodo",
    "qodo-merge-pro[bot]": "qodo",
    "ellipsis-dev[bot]": "ellipsis",
    "sourcery-ai[bot]": "sourcery",
    "deepsource-autofix[bot]": "deepsource",
    "greptile-apps[bot]": "greptile",
}

# AI tools seen reviewing PRs. Recorded on PRs as reviewers, never as an
# authorship label (a reviewer comment is not authorship).
REVIEW_BOT_LOGINS: Dict[str, str] = {
    "coderabbitai[bot]": "CodeRabbit",
    "coderabbitai": "CodeRabbit",
    "sourcery-ai[bot]": "Sourcery",
    "gemini-code-assist[bot]": "Gemini Code Assist",
    "copilot-pull-request-reviewer[bot]": "Copilot review",
    "copilot-pull-request-reviewer": "Copilot review",
    "cursor[bot]": "Cursor Bugbot",
    "chatgpt-codex-connector[bot]": "Codex review",
    "qodo-merge-pro[bot]": "Qodo",
    "qodo-code-review[bot]": "Qodo",
    "greptile-apps[bot]": "Greptile",
    "ellipsis-dev[bot]": "Ellipsis",
    "deepsource-autofix[bot]": "DeepSource",
    "claude[bot]": "Claude review",
}

# Automation that is not a coding agent (dependency bumps, CI, releases).
CI_BOT_LOGINS = {
    "github-actions[bot]", "github-actions", "dependabot[bot]", "dependabot-preview[bot]",
    "renovate[bot]", "snyk-bot", "pre-commit-ci[bot]", "codecov[bot]", "allcontributors[bot]",
    "imgbot[bot]", "mergify[bot]", "stale[bot]", "release-please[bot]", "semantic-release-bot",
    "kodiakhq[bot]", "microsoft-github-policy-service[bot]", "dotnet-maestro[bot]",
    "stainless-app[bot]", "octo-sts[bot]", "fern-api[bot]", "gcf-owl-bot[bot]",
}

_NOREPLY_RE = re.compile(r"^(?:(?P<id>\d+)\+)?(?P<login>[^@]+)@users\.noreply\.github\.com$", re.I)


def login_from_noreply(email: Optional[str]) -> Optional[str]:
    """`123+copilot-swe-agent[bot]@users.noreply.github.com` -> the login."""
    if not email:
        return None
    m = _NOREPLY_RE.match(email.strip())
    return m.group("login").lower() if m else None


def brand_for_login(login: Optional[str]) -> Optional[str]:
    if not login:
        return None
    return keep(AGENT_LOGINS.get(login.strip().lower()))


def brand_for_email(email: Optional[str], name: Optional[str] = None) -> Optional[str]:
    """Agent brand for an author/committer/co-author identity. The email
    decides, except for tools that reuse another agent's address
    (TOOL_NAME_OVERRIDES) and addresses shared with non-agent tooling
    (NAME_EMAIL_PAIRS), where the name must match too."""
    if not email:
        return None
    email_l = email.strip().lower()
    name_l = (name or "").strip().lower()
    if name_l in TOOL_NAME_OVERRIDES and email_l in AGENT_EMAILS:
        return keep(TOOL_NAME_OVERRIDES[name_l])
    if email_l in AGENT_EMAILS:
        return keep(AGENT_EMAILS[email_l])
    for name_re, pair_email, brand in NAME_EMAIL_PAIRS:
        if email_l == pair_email and name_re.search(name_l):
            return keep(brand)
    return brand_for_login(login_from_noreply(email_l))


def is_ci_bot(login: Optional[str]) -> bool:
    return bool(login) and login.strip().lower() in CI_BOT_LOGINS


def _compile(table: Dict[str, List[str]], flags: int = re.I) -> List[Tuple[str, Pattern[str]]]:
    return [(brand, re.compile(p, flags)) for brand, pats in table.items() for p in pats]


# ---------------------------------------------------------------------------
# Trailer co-author names (used only when the trailer's e-mail is unknown)
# ---------------------------------------------------------------------------

TRAILER_NAME_PATTERNS = _compile({
    "claude_code": [r"^claude(?: code)?$", r"^claude (?:opus|sonnet|haiku|fable|mythos)\b", r"^claude \(",
                    r"^claude(?: code)?\s*:"],
    "copilot": [r"^(?:github )?copilot$"],
    "cursor": [r"^cursor(?: agent)?$"],
    "codex": [r"^codex\b", r"^chatgpt codex\b", r"^openai codex\b"],
    "gemini": [r"^gemini(?: code assist| cli)?$", r"^gemini \d", r"^gemini(?: cli)?\s*:"],
    "antigravity": [r"^antigravity(?: cli| agent)?$"],
    "aider": [r"^aider$", r"^aider \("],
    "devin": [r"^devin ai$"],
    "junie": [r"^junie$"],
    "openhands": [r"^openhands(?:-agent)?$"],
    "qwen_coder": [r"^qwen[- ]coder$", r"^qwen code$"],
    "kiro": [r"^kiro(?: agent| ide)?$"],
    "replit": [r"^replit agent$"],
    "lovable": [r"^lovable$"],
    "grok": [r"^grok \d"],
    "amp": [r"^amp$"],
    "opencode": [r"^opencode(?: agent)?$", r"^opencode \("],
    "crush": [r"^crush$", r"\bvia crush$"],
    "cline": [r"^cline$"],
    "letta": [r"^letta code$"],
    "mistral_vibe": [r"^mistral vibe$"],
    "trae": [r"^trae(?: cli)?$"],
    "kimi": [r"^kimi code$"],
    "windsurf": [r"^windsurf(?: cascade)?$"],
    "tabnine": [r"^tabnine(?: cli)?$"],
})

# Trailer keys whose value names a co-author or the tool that helped.
TRAILER_RE = re.compile(
    r"^\s*(?P<key>co-authored-by|generated-by|assisted-by|ai-assisted-by)\s*:\s*"
    r"(?P<name>[^<\n]*?)\s*(?:<(?P<email>[^>\n]*)>)?\s*$",
    re.I | re.M,
)

# Trailers that a specific tool writes and that are a signal on their own
# (key -> (value regex, brand)). Checked line by line.
TOOL_TRAILERS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"^\s*agent-logs-url:\s*https://github\.com/[^/\s]+/[^/\s]+/sessions/\S+", re.I | re.M), "copilot"),
    (re.compile(r"^\s*claude-session:\s*https://claude\.ai/code/session_\S+", re.I | re.M), "claude_code"),
    (re.compile(r"^\s*amp-thread-id:\s*https://ampcode\.com/threads/\S+", re.I | re.M), "amp"),
    (re.compile(r"^\s*replit-commit-author:\s*(?:agent|assistant)\s*$", re.I | re.M), "replit"),
    (re.compile(r"^\s*x-lovable-edit-id:\s*\S+", re.I | re.M), "lovable"),
    (re.compile(r"^\s*made-with:\s*cursor\s*$", re.I | re.M), "cursor"),
]

# ---------------------------------------------------------------------------
# Markers the agents write into commit messages and PR bodies
# ---------------------------------------------------------------------------

BODY_MARKER_PATTERNS = _compile({
    "claude_code": [r"generated with \[?claude code\]?", r"https?://claude\.(?:ai/code|com/claude-code)"],
    "codex": [r"chatgpt\.com/codex/tasks/"],
    "copilot": [r"copilot-workspace\.githubnext\.com"],
    "devin": [r"app\.devin\.ai/sessions/"],
    "cursor": [r"cursor\.com/background-agent", r"cursor\.com/agents\b", r"made with \[cursor\]\(https://cursor\.com\)",
               r"generated with \[cursor\]\(https://cursor\.com\)"],
    "amp": [r"ampcode\.com/threads/"],
    "opencode": [r"generated with \[?opencode\]?"],
    "gemini": [r"generated with \[?gemini cli\]?"],
    "antigravity": [r"generated with \[?antigravity(?: cli)?\]?"],
    "qwen_coder": [r"generated with \[?qwen code\]?"],
    "cline": [r"generated with \[cline\]\(https://cline\.bot\)"],
    "windsurf": [r"generated with \[?windsurf(?: cascade)?\]?"],
    "crush": [r"generated with crush\b"],
    "jules": [r"jules\.google(?:\.com)?/(?:task|session)"],
})

# A commit subject the tool itself prefixes (classic Aider behaviour).
SUBJECT_PREFIX_PATTERNS = _compile({
    "aider": [r"^aider: "],
})

# `Jane Doe (aider)`: aider's default author/committer name suffix.
AIDER_NAME_RE = re.compile(r"\(aider\)\s*$", re.I)

# ---------------------------------------------------------------------------
# @mentions that start an agent from an issue or PR comment
# ---------------------------------------------------------------------------

MENTIONS: Dict[str, str] = {
    "claude": "claude_code",
    "codex": "codex",
    "copilot": "copilot",
    "cursor": "cursor",
    "jules": "jules",
    "devin": "devin",
    "devin-ai-integration": "devin",
    "gemini-cli": "gemini",
    "gemini-code-assist": "gemini",
    "openhands": "openhands",
    "openhands-agent": "openhands",
    "coderabbitai": "coderabbit",
    "sweep-ai": "sweep",
    "codegen-sh": "codegen",
    "kiro-agent": "kiro",
    "ellipsis-dev": "ellipsis",
}
MENTION_RE = re.compile(
    r"(?<![\w.@/`-])@(" + "|".join(sorted(map(re.escape, MENTIONS), key=len, reverse=True)) + r")(?![\w-])",
    re.I,
)


def mentioned_brands(text: Optional[str]) -> List[str]:
    """Agents @mentioned in a comment, issue or PR body (code spans and
    e-mail addresses excluded by the anchors)."""
    if not text:
        return []
    found: List[str] = []
    for m in MENTION_RE.finditer(text):
        brand = MENTIONS[m.group(1).lower()]
        if is_active(brand) and brand not in found:
            found.append(brand)
    return found


# ---------------------------------------------------------------------------
# PR head-branch prefixes and exact PR labels
# ---------------------------------------------------------------------------

BRANCH_PREFIXES: Dict[str, str] = {
    "claude/": "claude_code",
    "claudecode/": "claude_code",
    "copilot/": "copilot",
    "codex/": "codex",
    "cursor/": "cursor",
    "devin/": "devin",
    "jules/": "jules",
    "openhands/": "openhands",
    "opencode/": "opencode",
    "sweep/": "sweep",
    "goose/": "goose",
    "open-swe/": "open_swe",
    "codegen-bot/": "codegen",
    "windsurf-bot/": "windsurf",
    "gru/": "gru",
    "cline/": "cline",
    "q-dev-issue-": "amazon_q",
}
BRANCH_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"^jules-\d{6,}-[0-9a-f]{6,}$", re.I), "jules"),
    (re.compile(r"^jetbrains_junie", re.I), "junie"),
]

PR_LABELS: Dict[str, str] = {
    "codex": "codex",
    "ai-generated": "unknown",
    "copilot-generated": "copilot",
    "claude-code": "claude_code",
    "amp": "amp",
    "open-swe": "open_swe",
}


def brand_for_branch(branch: Optional[str]) -> Optional[str]:
    if not branch:
        return None
    b = branch.strip().lower()
    for prefix, brand in BRANCH_PREFIXES.items():
        if b.startswith(prefix):
            return keep(brand)
    for pattern, brand in BRANCH_PATTERNS:
        if pattern.search(b):
            return keep(brand)
    return None


def brand_for_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return keep(PR_LABELS.get(label.strip().lower()))


# ---------------------------------------------------------------------------
# Repo file markers (configuration, rules and guidance an agent reads)
# ---------------------------------------------------------------------------

# (path or directory prefix ending in "/", brand). Matched against
# repo-relative paths, case-sensitive, from the repo root unless the entry
# starts with "**/" (any depth).
FILE_MARKERS: List[Tuple[str, str]] = [
    ("CLAUDE.md", "claude_code"), ("**/CLAUDE.md", "claude_code"), (".claude/", "claude_code"),
    (".claudeignore", "claude_code"), (".github/workflows/claude.yml", "claude_code"),
    (".github/workflows/claude-code-review.yml", "claude_code"),
    (".github/copilot-instructions.md", "copilot"), (".github/instructions/", "copilot"),
    (".github/prompts/", "copilot"), (".github/agents/", "copilot"), (".github/chatmodes/", "copilot"),
    (".github/workflows/copilot-setup-steps.yml", "copilot"), (".copilotignore", "copilot"),
    (".cursorrules", "cursor"), (".cursor/", "cursor"), ("CURSOR.md", "cursor"),
    (".cursorignore", "cursor"), (".cursorindexingignore", "cursor"),
    (".codex/", "codex"),
    ("GEMINI.md", "gemini"), ("**/GEMINI.md", "gemini"), (".gemini/", "gemini"),
    ("AGENTS.md", "unknown"), ("**/AGENTS.md", "unknown"),
    (".aider.conf.yml", "aider"), (".aider.conf.yaml", "aider"), (".aiderignore", "aider"),
    (".clinerules", "cline"), (".cline/", "cline"), ("memory-bank/", "cline"), ("memory_bank/", "cline"),
    (".roo/", "roo_code"), (".roorules", "roo_code"),
    (".windsurfrules", "windsurf"), (".windsurf/", "windsurf"), (".codeium/", "windsurf"),
    (".junie/", "junie"), (".junieignore", "junie"),
    (".jules/", "jules"), (".devin/", "devin"),
    (".openhands/", "openhands"), (".openhands_instructions", "openhands"),
    (".goosehints", "goose"), (".continue/", "continue"),
    (".kiro/", "kiro"), (".kilocode/", "kilo_code"), (".kilocodeignore", "kilo_code"),
    (".amazonq/", "amazon_q"), ("AmazonQ.md", "amazon_q"), ("AMAZONQ.md", "amazon_q"),
    (".trae/", "trae"), (".traerules", "trae"), (".marscode/", "trae"),
    ("WARP.md", "warp"), (".warp/", "warp"),
    ("CRUSH.md", "crush"), (".crush/", "crush"),
    ("QWEN.md", "qwen_coder"),
    (".opencode/", "opencode"), ("opencode.json", "opencode"),
    ("AGENT.md", "amp"), (".sourcegraph/", "amp"),
    (".factory/", "factory"),
    (".augment/", "augment"), (".augmentignore", "augment"), (".augment-guidelines", "augment"),
    (".charlie/", "charlie"), (".kimi/", "kimi"), (".pi/", "pi"), (".rovodev/", "rovo_dev"),
    ("CODEBUDDY.md", "codebuddy"), (".lingma/", "lingma"), (".comate/", "comate"),
    (".brokk/", "brokk"), (".plandex/", "plandex"), (".plandexignore", "plandex"),
    (".qodo/", "qodo"), (".verdent/", "verdent"), (".serena/", "serena"),
    (".tessl/", "tessl"), (".rulesync/", "rulesync"),
    (".specify/", "speckit"), (".taskmaster/", "taskmaster"), ("README-task-master.md", "taskmaster"),
    ("replit.md", "replit"), (".bolt/", "bolt"),
]


def file_marker_brand(path: str) -> Optional[Tuple[str, str]]:
    """Return (marker, brand) if a repo-relative path is an agent marker."""
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    base = p.rsplit("/", 1)[-1]
    for marker, brand in DETECT_FILE_MARKERS:
        if marker.startswith("**/"):
            if base == marker[3:] and "/" in p:
                return marker, brand
        elif marker.endswith("/"):
            if p.startswith(marker):
                return marker, brand
        elif p == marker:
            return marker, brand
    return None


# ---------------------------------------------------------------------------
# Restricting detection to a few brands
# ---------------------------------------------------------------------------

# None = every brand. set_active_brands() narrows detection: the pattern
# lists the detectors loop over are pruned to the active brands, and
# identity lookups (addresses, logins) still use the full tables -- so a
# tool that reuses another agent's address (Kimi Code writing as
# noreply@anthropic.com) resolves to its own brand and is then dropped,
# rather than being counted as the agent whose address it borrowed. The
# full tables also stay in use for stripping signals out of commit
# messages and for telling bot accounts from people.
ACTIVE_BRANDS: Optional[frozenset] = None

DETECT_TRAILER_NAMES = TRAILER_NAME_PATTERNS
DETECT_TOOL_TRAILERS = TOOL_TRAILERS
DETECT_BODY_MARKERS = BODY_MARKER_PATTERNS
DETECT_SUBJECT_PREFIXES = SUBJECT_PREFIX_PATTERNS
DETECT_FILE_MARKERS = FILE_MARKERS


def is_active(brand: Optional[str]) -> bool:
    return brand is not None and (ACTIVE_BRANDS is None or brand in ACTIVE_BRANDS)


def keep(brand: Optional[str]) -> Optional[str]:
    """`brand` if it is being detected, else None."""
    return brand if is_active(brand) else None


def set_active_brands(brands: Optional[Iterable[str]]) -> None:
    """Detect only `brands` (None = all). Unknown brand ids raise."""
    global ACTIVE_BRANDS, DETECT_TRAILER_NAMES, DETECT_TOOL_TRAILERS, DETECT_BODY_MARKERS, \
        DETECT_SUBJECT_PREFIXES, DETECT_FILE_MARKERS
    if brands is None:
        ACTIVE_BRANDS = None
    else:
        chosen = frozenset(brands)
        unknown = sorted(chosen - set(BRANDS))
        if unknown:
            raise ValueError(f"unknown agent id(s): {', '.join(unknown)}")
        ACTIVE_BRANDS = chosen
    DETECT_TRAILER_NAMES = [(b, p) for b, p in TRAILER_NAME_PATTERNS if is_active(b)]
    DETECT_TOOL_TRAILERS = [(p, b) for p, b in TOOL_TRAILERS if is_active(b)]
    DETECT_BODY_MARKERS = [(b, p) for b, p in BODY_MARKER_PATTERNS if is_active(b)]
    DETECT_SUBJECT_PREFIXES = [(b, p) for b, p in SUBJECT_PREFIX_PATTERNS if is_active(b)]
    DETECT_FILE_MARKERS = [(m, b) for m, b in FILE_MARKERS if is_active(b)]


# Only the top five agents are detected. Tests switch to every brand
# (set_active_brands(None)) to check the full tables.
set_active_brands(TOP_BRANDS)
