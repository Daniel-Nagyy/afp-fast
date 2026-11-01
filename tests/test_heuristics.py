from afp import config
from afp.heuristics import (
    detect_trailer, detect_bot_account, detect_email_domain,
    detect_pr_marker, resolve_label,
)


def test_detect_trailer_claude():
    sig = detect_trailer("Fix bug\n\nCo-authored-by: Claude <noreply@anthropic.com>")
    assert sig.has_trailer
    assert sig.agent_brand == "claude_code"
    assert sig.label_source == "trailer"
    assert sig.label == config.LABEL_AGENT


def test_detect_trailer_unrelated_coauthor_is_unlabeled():
    sig = detect_trailer("Fix bug\n\nCo-authored-by: Jane Doe <jane@example.com>")
    assert sig.has_trailer  # trailer exists...
    assert sig.label == config.LABEL_UNLABELED  # ...but doesn't match a known brand


def test_detect_trailer_no_trailer():
    sig = detect_trailer("Just a normal commit message")
    assert not sig.has_trailer
    assert sig.label == config.LABEL_UNLABELED


def test_detect_bot_account_known_bot():
    sig = detect_bot_account("copilot-swe-agent[bot]")
    assert sig.is_bot_account
    assert sig.agent_brand == "copilot"
    assert sig.label == config.LABEL_AGENT


def test_detect_bot_account_ci_bot_not_counted_as_agent():
    sig = detect_bot_account("github-actions[bot]")
    assert sig.is_bot_account
    assert sig.agent_brand is None
    assert sig.label == config.LABEL_UNLABELED


def test_detect_bot_account_unknown_login():
    sig = detect_bot_account("some-random-user")
    assert not sig.is_bot_account
    assert sig.label == config.LABEL_UNLABELED


def test_detect_email_domain_codex():
    sig = detect_email_domain("noreply@openai.com")
    assert sig.email_domain_match
    assert sig.agent_brand == "codex"
    assert sig.label == config.LABEL_AGENT


def test_detect_email_domain_generic_github_not_used_alone():
    sig = detect_email_domain("123456+someuser@github.com")
    # github.com maps to None (too generic) -- must NOT be treated as a signal
    assert sig.label == config.LABEL_UNLABELED


def test_detect_pr_marker_cursor():
    sig = detect_pr_marker(title="Fix via Cursor agent", branch_name="cursor/fix-123")
    assert sig.pr_marker_match
    assert sig.agent_brand == "cursor"
    assert sig.label == config.LABEL_AGENT


def test_resolve_label_priority_trailer_over_pr_marker():
    """Per the methodology's label-source priority, a trailer match wins
    even if the PR text happens to mention a different brand."""
    sig = resolve_label(
        commit_message="Fix\n\nCo-authored-by: Claude <noreply@anthropic.com>",
        pr_title="Automated fix via Copilot",
    )
    assert sig.label_source == "trailer"
    assert sig.agent_brand == "claude_code"


def test_resolve_label_no_signal_is_unlabeled_not_human():
    sig = resolve_label(commit_message="just a normal commit")
    assert sig.label == config.LABEL_UNLABELED
    assert sig.label != "human"  # the label vocabulary must never include "human"
