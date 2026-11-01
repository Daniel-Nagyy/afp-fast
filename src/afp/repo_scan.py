"""
Repo-level adoption signals beyond rules/guidance files.

Works from two inputs: the list of paths at HEAD (the tree listing) and
the text of a few small files (fetched in one GraphQL query). wanted_paths() says which files to
read; scan() turns paths + contents into RepoSignal records:

  workflow            a GitHub Actions workflow runs an agent
  mcp                 Model Context Protocol config for an agent
  dependency          the agent's package is a dependency
  devcontainer        a dev container installs the agent or its extension
  gitignore           agent files are kept out of git (hidden adoption)
  hidden_attribution  a setting or instruction switches commit attribution off
  agent_dir           sub-agents, commands, skills or prompts for an agent
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .schema import RepoSignal
from .signatures import is_active

# --- workflows --------------------------------------------------------------

WORKFLOW_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"anthropics/claude-code(?:-base)?-action", re.I), "claude_code", "anthropics/claude-code-action"),
    (re.compile(r"@anthropic-ai/claude-code\b", re.I), "claude_code", "installs @anthropic-ai/claude-code"),
    (re.compile(r"openai/codex-action", re.I), "codex", "openai/codex-action"),
    (re.compile(r"@openai/codex\b", re.I), "codex", "installs @openai/codex"),
    (re.compile(r"google-github-actions/run-gemini-cli", re.I), "gemini", "google-github-actions/run-gemini-cli"),
    (re.compile(r"@google/gemini-cli\b", re.I), "gemini", "installs @google/gemini-cli"),
    (re.compile(r"\bcursor-agent\b|cursor\.com/install", re.I), "cursor", "cursor-agent CLI"),
    (re.compile(r"all-hands-ai/openhands|openhands-resolver", re.I), "openhands", "OpenHands resolver"),
    (re.compile(r"\baider-chat\b|aider-ai/aider|paul-gauthier/aider", re.I), "aider", "runs aider"),
    (re.compile(r"sst/opencode|\bopencode-ai\b", re.I), "opencode", "runs opencode"),
    (re.compile(r"coderabbitai/", re.I), "coderabbit", "CodeRabbit action"),
]
WORKFLOW_FILENAMES: Dict[str, str] = {
    "copilot-setup-steps.yml": "copilot",
    "copilot-setup-steps.yaml": "copilot",
}

# --- MCP ----------------------------------------------------------------------

MCP_FILES: Dict[str, str] = {
    ".mcp.json": "claude_code",          # Claude Code's project MCP file (also read by others)
    ".cursor/mcp.json": "cursor",
    ".vscode/mcp.json": "copilot",
    ".roo/mcp.json": "roo_code",
    ".kiro/settings/mcp.json": "kiro",
    ".amazonq/mcp.json": "amazon_q",
    ".gemini/settings.json": "gemini",   # only if it contains mcpServers
}

# --- dependencies ---------------------------------------------------------

NPM_PACKAGES: Dict[str, str] = {
    "@anthropic-ai/claude-code": "claude_code",
    "@openai/codex": "codex",
    "@google/gemini-cli": "gemini",
    "@augmentcode/auggie": "augment",
    "@sourcegraph/amp": "amp",
    "opencode-ai": "opencode",
    "@qwen-code/qwen-code": "qwen_coder",
    "@charmland/crush": "crush",
    "task-master-ai": "taskmaster",
    "@github/copilot": "copilot",
}
PY_PACKAGES: Dict[str, str] = {
    "aider-chat": "aider",
    "openhands-ai": "openhands",
}
DEPENDENCY_FILES = ("package.json", "requirements.txt", "requirements-dev.txt", "pyproject.toml")

# --- dev containers -------------------------------------------------------

DEVCONTAINER_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"features/claude-code\b|anthropic\.claude-code", re.I), "claude_code", "Claude Code feature/extension"),
    (re.compile(r"features/codex\b|openai\.chatgpt", re.I), "codex", "Codex feature/extension"),
    (re.compile(r"features/gemini-cli\b|google\.geminicodeassist", re.I), "gemini", "Gemini feature/extension"),
    (re.compile(r"saoudrizwan\.claude-dev", re.I), "cline", "Cline extension"),
    (re.compile(r"rooveterinaryinc\.roo-cline", re.I), "roo_code", "Roo Code extension"),
    (re.compile(r"continue\.continue", re.I), "continue", "Continue extension"),
    (re.compile(r"kilocode\.kilo-code", re.I), "kilo_code", "Kilo Code extension"),
    (re.compile(r"augment\.vscode-augment", re.I), "augment", "Augment extension"),
    (re.compile(r"github\.copilot-chat", re.I), "copilot", "Copilot Chat extension"),
]
DEVCONTAINER_FILES = (".devcontainer/devcontainer.json", ".devcontainer.json", ".vscode/extensions.json")

# --- .gitignore ----------------------------------------------------------------

GITIGNORE_ENTRIES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^/?(?:\*\*/)?CLAUDE(?:\.local)?\.md$|^/?\.claude(?:/|$)", re.I), "claude_code"),
    (re.compile(r"^/?\.cursor(?:/|rules$|ignore$|$)", re.I), "cursor"),
    (re.compile(r"^/?\.aider", re.I), "aider"),
    (re.compile(r"^/?(?:GEMINI\.md|\.gemini(?:/|$))", re.I), "gemini"),
    (re.compile(r"^/?\.codex(?:/|$)", re.I), "codex"),
    (re.compile(r"^/?\.github/copilot-instructions\.md$", re.I), "copilot"),
    (re.compile(r"^/?\.windsurf(?:/|rules$|$)", re.I), "windsurf"),
    (re.compile(r"^/?\.(?:clinerules|cline)(?:/|$)", re.I), "cline"),
    (re.compile(r"^/?\.roo(?:/|rules$|$)", re.I), "roo_code"),
    (re.compile(r"^/?\.junie(?:/|$)", re.I), "junie"),
    (re.compile(r"^/?\.kiro(?:/|$)", re.I), "kiro"),
    (re.compile(r"^/?\.(?:opencode|crush|continue|kilocode|trae|amazonq|augment)(?:/|$)", re.I), "unknown"),
    (re.compile(r"^/?AGENTS\.md$", re.I), "unknown"),
    (re.compile(r"^/?\.specstory(?:/|$)", re.I), "unknown"),
]

# --- hidden attribution ---------------------------------------------------------

NO_ATTRIBUTION_RE = re.compile(
    r"(?:do not|don't|never|no|without)\b[^\n]{0,60}\b(?:co-?authored|attribution|generated with|ai[- ]generated|claude as)\b",
    re.I)
# guidance file -> the agent that reads it (AGENTS.md is shared by several)
GUIDANCE_FILES: Dict[str, str] = {
    "CLAUDE.md": "claude_code", "AGENTS.md": "unknown", "GEMINI.md": "gemini", ".cursorrules": "cursor",
    ".github/copilot-instructions.md": "copilot",
}

# files read only for one agent's signals (the rest serve several)
_FILE_BRANDS: Dict[str, str] = {
    "requirements.txt": "", "requirements-dev.txt": "", "pyproject.toml": "",   # python agents: aider, openhands
    ".gemini/settings.json": "gemini", ".aider.conf.yml": "aider",
}
_PY_AGENTS = set(PY_PACKAGES.values())

# --- agent sub-folders ----------------------------------------------------------

AGENT_DIRS: Dict[str, str] = {
    ".claude/agents/": "claude_code", ".claude/commands/": "claude_code", ".claude/skills/": "claude_code",
    ".claude/hooks/": "claude_code",
    ".github/agents/": "copilot", ".github/prompts/": "copilot", ".github/chatmodes/": "copilot",
    ".github/instructions/": "copilot",
    ".gemini/commands/": "gemini", ".cursor/commands/": "cursor", ".cursor/rules/": "cursor",
    ".opencode/agent/": "opencode", ".opencode/command/": "opencode",
    ".kiro/steering/": "kiro", ".kiro/specs/": "kiro", ".roo/rules/": "roo_code",
    ".factory/droids/": "factory", ".factory/commands/": "factory",
}


def wanted_paths(paths: Iterable[str]) -> List[str]:
    """The small files scan() needs the text of (at most ~40)."""
    present = set(paths)
    want = [p for p in present if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml"))][:25]
    for p in candidate_files():
        if p in present:
            want.append(p)
    return sorted(set(want))


def candidate_files() -> List[str]:
    """Fixed paths scan() may read, minus those that only matter for
    agents not being detected (see signatures.set_active_brands)."""
    out = []
    for p in list(DEPENDENCY_FILES) + list(DEVCONTAINER_FILES) + list(GUIDANCE_FILES) + [
            ".gitignore", ".claude/settings.json", ".gemini/settings.json", ".aider.conf.yml"]:
        brand = _FILE_BRANDS.get(p)
        if brand == "" and not any(is_active(b) for b in _PY_AGENTS):
            continue
        if brand and not is_active(brand):
            continue
        if p in GUIDANCE_FILES and not is_active(GUIDANCE_FILES[p]):
            continue
        out.append(p)
    return out


def _deps_in(path: str, text: str) -> Iterable[str]:
    if path == "package.json":
        try:
            doc = json.loads(text)
        except ValueError:
            return []
        names = set()
        for key in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
            names |= set((doc.get(key) or {}).keys())
        return names
    low = text.lower()
    return [pkg for pkg in PY_PACKAGES if re.search(rf"(?<![\w-]){re.escape(pkg)}(?![\w-])", low)]


def scan(paths: List[str], read: Callable[[str], Optional[str]],
         added: Optional[Callable[[str], Optional[str]]] = None) -> List[RepoSignal]:
    """All repo signals for one repo. `read(path)` returns file text or
    None; `added(path)` optionally returns the date a file was first
    added (not available from the API, so the pipeline leaves it out)."""
    signals: List[RepoSignal] = []
    present = set(paths)

    def emit(kind, brand, path, detail, strength="medium"):
        if not is_active(brand):
            return
        for s in signals:
            if (s.kind, s.brand, s.path) == (kind, brand, path):
                return
        signals.append(RepoSignal(kind=kind, brand=brand, path=path, detail=detail, strength=strength,
                                  first_added_at=added(path) if added and kind in ("workflow", "mcp") else None))

    # workflows
    for p in sorted(p for p in present if p.startswith(".github/workflows/")):
        base = p.rsplit("/", 1)[-1]
        if base in WORKFLOW_FILENAMES:
            emit("workflow", WORKFLOW_FILENAMES[base], p, base, "strong")
        text = read(p) if p.endswith((".yml", ".yaml")) else None
        for pattern, brand, detail in WORKFLOW_PATTERNS:
            if text and pattern.search(text):
                emit("workflow", brand, p, detail, "strong")

    # MCP
    for p, brand in MCP_FILES.items():
        if p not in present:
            continue
        if p == ".gemini/settings.json":
            text = read(p) or ""
            if "mcpServers" not in text:
                continue
        emit("mcp", brand if p != ".mcp.json" else "claude_code", p, "MCP server config")

    # dependencies
    for p in DEPENDENCY_FILES:
        if p in present:
            text = read(p)
            if not text:
                continue
            for dep in _deps_in(p, text):
                brand = NPM_PACKAGES.get(dep) or PY_PACKAGES.get(dep)
                if brand:
                    emit("dependency", brand, p, dep, "weak")

    # dev containers and recommended extensions
    for p in DEVCONTAINER_FILES:
        if p in present:
            text = read(p) or ""
            for pattern, brand, detail in DEVCONTAINER_PATTERNS:
                if pattern.search(text):
                    emit("devcontainer", brand, p, detail, "weak" if p.endswith("extensions.json") else "medium")

    # .gitignore entries hiding agent files
    if ".gitignore" in present:
        text = read(".gitignore") or ""
        hits = []
        for raw in text.splitlines():
            entry = raw.strip()
            if not entry or entry.startswith("#") or entry.startswith("!"):
                continue
            for pattern, brand in GITIGNORE_ENTRIES:
                if pattern.search(entry):
                    hits.append((brand, entry))
                    break
        # many different agents at once looks like a copied template (e.g. github/gitignore's Agents.gitignore)
        templated = len({b for b, _ in hits}) >= 5
        for brand, entry in hits:
            emit("gitignore", brand, ".gitignore", entry + (" (template-like)" if templated else ""),
                 "weak" if templated else "medium")

    # attribution switched off
    settings = read(".claude/settings.json") if ".claude/settings.json" in present else None
    if settings:
        try:
            doc = json.loads(settings)
        except ValueError:
            doc = {}
        attribution = doc.get("attribution")
        if doc.get("includeCoAuthoredBy") is False or doc.get("gitAttribution") is False or \
                (isinstance(attribution, dict) and attribution.get("commit") in ("", False)):
            emit("hidden_attribution", "claude_code", ".claude/settings.json", "commit attribution disabled", "strong")
    aider_conf = read(".aider.conf.yml") if ".aider.conf.yml" in present else None
    if aider_conf and re.search(r"^\s*attribute-(?:author|committer)\s*:\s*false", aider_conf, re.I | re.M):
        emit("hidden_attribution", "aider", ".aider.conf.yml", "(aider) name attribution disabled", "strong")
    for p, brand in GUIDANCE_FILES.items():
        if p in present:
            text = read(p) or ""
            m = NO_ATTRIBUTION_RE.search(text)
            if m:
                emit("hidden_attribution", brand, p, m.group(0)[:120], "medium")

    # agent sub-folders
    for prefix, brand in AGENT_DIRS.items():
        n = sum(1 for p in present if p.startswith(prefix))
        if n:
            emit("agent_dir", brand, prefix, f"{n} file(s)", "medium")

    return signals
