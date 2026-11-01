"""Regression tests for the signature roster.

The "must NOT match" cases are real commit messages, co-authors and
logins from a 185-repo corpus run where free-text brand matching
produced false positives (Vim plugins "using Cursor", Aider detected in
the login "osman-haider"); the "must match" cases are real agent traces
from the same run.
"""

import pytest

from afp import config
from afp.features import strip_label_signals
from afp.heuristics import (apply_pr_label, detect_bot_account, detect_email_domain,
                            detect_message_marker, detect_pr_marker, resolve_label)
from afp.schema import LabelSignal
from afp.signatures import brand_for_email, file_marker_brand, login_from_noreply

# these tests cover the full signature tables, not just the top five
pytestmark = pytest.mark.usefixtures("all_agents")

AGENT = config.LABEL_AGENT
UNLABELED = config.LABEL_UNLABELED


@pytest.mark.parametrize("message", [
    "Move cursor to the next line after include in C",
    "Save current cursor position to jumplist before MoveToParentHeader",
    "Add aliases for cursorline/cursorcolumn maps",
    "runtest: respond to cursor query escape sequence",
    "Use the Claude API client for summaries",
    "Port the codex lookup table to Rust",
    "Thanks to the Gemini team for the report",
    "Fix bug\n\nCo-authored-by: Claude Martin <claude.martin@example.com>",
    "Fix bug\n\nCo-authored-by: Cursor Smith <csmith@example.com>",
    "Release\n\nCo-authored-by: Jules Verne <jules@example.com>",
])
def test_brand_words_in_ordinary_messages_are_not_signals(message):
    assert resolve_label(commit_message=message).label == UNLABELED


@pytest.mark.parametrize("message,brand,source", [
    ("Fix\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>", "claude_code", "trailer"),
    ("Fix\n\nCo-authored-by: Claude <noreply@anthropic.com>", "claude_code", "trailer"),
    ("Fix #2310\n\nCo-authored-by: Cursor <cursoragent@cursor.com>", "cursor", "trailer"),
    ("Fix\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>", "cursor", "trailer"),
    ("s\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
     "copilot", "trailer"),
    ("Update psbt.ts\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
     "copilot", "trailer"),
    ("Apply suggestion\n\nCo-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>",
     "gemini", "trailer"),
    ("Add license\n\n\U0001F916 Generated with [Claude Code](https://claude.com/claude-code)",
     "claude_code", "message_marker"),
    ("Fix\n\nFor more details, open the [Copilot Workspace session](https://copilot-workspace.githubnext.com/a/b)",
     "copilot", "message_marker"),
    ("aider: Add retry to the HTTP client", "aider", "message_marker"),
])
def test_real_agent_traces_are_labeled(message, brand, source):
    sig = resolve_label(commit_message=message)
    assert sig.label == AGENT
    assert sig.agent_brand == brand
    assert sig.label_source == source


def test_trailer_and_marker_both_fire_and_trailer_wins():
    msg = ("Add tests\n\n\U0001F916 Generated with [Claude Code](https://claude.ai/code)\n\n"
           "Co-Authored-By: Claude <noreply@anthropic.com>")
    sig = resolve_label(commit_message=msg)
    assert sig.label_source == "trailer"
    assert sig.message_marker_match and sig.has_trailer
    assert sig.brands == ["claude_code"]


@pytest.mark.parametrize("login", ["osman-haider", "farazhaider", "cruzjuniel", "clineil",
                                   "cursorai-vibecoder-cmyk", "claude", "jules"])
def test_human_logins_containing_brand_names_are_not_agents(login):
    assert detect_bot_account(login).label == UNLABELED


@pytest.mark.parametrize("login,brand", [
    ("copilot-swe-agent[bot]", "copilot"), ("Copilot", "copilot"),
    ("google-labs-jules[bot]", "jules"), ("chatgpt-codex-connector[bot]", "codex"),
    ("devin-ai-integration[bot]", "devin"), ("claude[bot]", "claude_code"),
])
def test_agent_bot_logins(login, brand):
    sig = detect_bot_account(login)
    assert sig.label == AGENT and sig.agent_brand == brand


def test_ci_bots_are_bots_but_not_agents():
    for login in ("dependabot[bot]", "renovate[bot]", "github-actions[bot]"):
        sig = detect_bot_account(login)
        assert sig.is_bot_account and sig.label == UNLABELED


def test_review_bots_are_review_kind_agents():
    """A commit co-authored by a review bot applied its AI fix: labelled,
    but kept apart from authoring agents by its kind."""
    from afp.signatures import BRAND_KIND
    sig = detect_bot_account("coderabbitai[bot]")
    assert sig.label == AGENT and sig.agent_brand == "coderabbit"
    assert BRAND_KIND["coderabbit"] == "review"


def test_noreply_email_resolves_login():
    assert login_from_noreply("198982749+Copilot@users.noreply.github.com") == "copilot"
    assert brand_for_email("198982749+Copilot@users.noreply.github.com") == "copilot"
    assert brand_for_email("123+octocat@users.noreply.github.com") is None


def test_company_domains_are_not_agent_addresses():
    assert detect_email_domain("someone@openai.com").label == UNLABELED
    assert detect_email_domain("someone@anthropic.com").label == UNLABELED
    assert detect_email_domain("noreply@anthropic.com").agent_brand == "claude_code"


@pytest.mark.parametrize("branch,brand", [
    ("codex/clear-stopped-process-logs", "codex"),
    ("copilot/fix-query-string-reordering", "copilot"),
    ("claude/fix-taskbar-freeze-9fnqS", "claude_code"),
    ("cursor/kotoba-runtime-b136", "cursor"),
    ("devin/1700000000-fix", "devin"),
])
def test_agent_branch_prefixes(branch, brand):
    sig = detect_pr_marker(title="anything", branch_name=branch)
    assert sig.label == AGENT and sig.agent_brand == brand


@pytest.mark.parametrize("title,branch,labels", [
    ("Fix cursor jumps during editing", "fix-cursor-jumps", []),
    ("Support Claude and Gemini models", "feature/llm-providers", ["enhancement"]),
    ("Copilot setup docs", "docs/copilot", []),       # prefix must be at the start
    ("Rename", "feature/codex/rename", []),
])
def test_pr_brand_words_without_markers_are_not_signals(title, branch, labels):
    assert detect_pr_marker(title=title, branch_name=branch, labels=labels).label == UNLABELED


def test_pr_body_marker_and_exact_label():
    body = "Implements retries.\n\nhttps://chatgpt.com/codex/tasks/task_e_123"
    assert detect_pr_marker(body=body).agent_brand == "codex"
    assert detect_pr_marker(labels=["codex"]).agent_brand == "codex"
    assert detect_pr_marker(labels=["Codex review wanted"]).label == UNLABELED


def test_message_marker_text_is_the_matched_marker():
    sig = detect_message_marker("x\n\nSee https://app.devin.ai/sessions/abc")
    assert sig.agent_brand == "devin"
    assert sig.message_marker_text == "app.devin.ai/sessions/"


@pytest.mark.parametrize("path,brand", [
    ("CLAUDE.md", "claude_code"), ("packages/api/CLAUDE.md", "claude_code"),
    (".claude/settings.json", "claude_code"), (".github/copilot-instructions.md", "copilot"),
    (".cursorrules", "cursor"), (".cursor/rules/style.mdc", "cursor"), ("GEMINI.md", "gemini"),
    ("AGENTS.md", "unknown"), (".junie/guidelines.md", "junie"), (".github/workflows/claude.yml", "claude_code"),
])
def test_file_markers(path, brand):
    assert file_marker_brand(path)[1] == brand


@pytest.mark.parametrize("path", ["src/cursor.py", "docs/claude.md", "lib/codex/table.c",
                                  "CLAUDE.md.bak", "test/.cursorrules.txt"])
def test_non_marker_paths(path):
    assert file_marker_brand(path) is None


def test_strip_label_signals_removes_agent_lines_keeps_human_trailers():
    msg = ("feat: add cache\n\nBody line.\n\n\U0001F916 Generated with [Claude Code](https://claude.com/claude-code)\n\n"
           "Co-Authored-By: Claude <noreply@anthropic.com>\n"
           "Co-authored-by: Jane Doe <jane@example.com>")
    assert strip_label_signals(msg) == "feat: add cache\n\nBody line.\n\n\nCo-authored-by: Jane Doe <jane@example.com>"
    assert strip_label_signals("aider: add retry") == "add retry"


def test_apply_pr_label_is_lowest_priority():
    pr = resolve_label(pr_branch_name="codex/x")
    unlabeled = LabelSignal()
    apply_pr_label(unlabeled, pr, 12)
    assert unlabeled.label == AGENT and unlabeled.label_source == "pr_marker"
    assert unlabeled.propagated_from_pr == 12

    trailer = resolve_label(commit_message="x\n\nCo-authored-by: Claude <noreply@anthropic.com>")
    apply_pr_label(trailer, pr, 13)
    assert trailer.label_source == "trailer" and trailer.agent_brand == "claude_code"
    assert trailer.brands == ["claude_code", "codex"]
