"""
Explicit-signal heuristics: trailer / message marker / bot account /
e-mail / PR marker detection, with brand resolution and label-source
priority.

This module ONLY looks for unambiguous, explicit signals. It never guesses
from structural/statistical properties -- that is features.py's job, and
keeping the two separate is what makes the "strip signals before training"
step in schema.CommitFeatureVector meaningful rather than circular.

Every pattern comes from signatures.py and is anchored to where the agent
writes it (a trailer line, a branch prefix, an exact login or address).
A brand name merely *mentioned* in a message, title or body is not a
signal: "fix cursor position" is not Cursor, and "Co-authored-by: Claude
Martin <claude.martin@example.com>" is not Claude Code.

Output label is always "agent_labeled" or "unlabeled" -- never "human".
Absence of a signal means "we don't know", not "this is human-written".
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from . import config
from . import signatures as sig_db
from .schema import LabelSignal


def _trailer_brand(name: str, email: str) -> Optional[str]:
    name = name.strip()
    brand = sig_db.brand_for_email(email, name)
    if brand:
        return brand
    for brand, pattern in sig_db.DETECT_TRAILER_NAMES:
        if pattern.search(name):
            return brand
    return None


def detect_trailer(commit_message: str) -> LabelSignal:
    """Look for a Co-authored-by / Generated-by / Assisted-by trailer
    naming a known agent by exact address or anchored name, then for the
    trailers a tool writes on its own (Agent-Logs-Url, Claude-Session,
    Amp-Thread-ID, Replit-Commit-Author: Agent, X-Lovable-Edit-ID,
    Made-with: Cursor)."""
    sig = LabelSignal()
    if not commit_message:
        return sig
    for m in sig_db.TRAILER_RE.finditer(commit_message):
        brand = _trailer_brand(m.group("name") or "", m.group("email") or "")
        if brand:
            sig.has_trailer = True
            sig.trailer_text = m.group(0).strip()
            sig.agent_brand = brand
            sig.label_source = "trailer"
            sig.label = config.LABEL_AGENT
            return sig
        # trailer exists but doesn't match a known brand -- keep looking,
        # but record that *a* trailer was present in case nothing else hits
        if not sig.has_trailer:
            sig.has_trailer = True
            sig.trailer_text = m.group(0).strip()
    for pattern, brand in sig_db.DETECT_TOOL_TRAILERS:
        m = pattern.search(commit_message)
        if m:
            sig.has_trailer = True
            sig.trailer_text = m.group(0).strip()
            sig.agent_brand = brand
            sig.label_source = "trailer"
            sig.label = config.LABEL_AGENT
            return sig
    return sig


def detect_message_marker(text: str) -> LabelSignal:
    """Look for a line an agent writes itself ("Generated with Claude
    Code", a Codex task link, an "aider: " subject prefix)."""
    sig = LabelSignal()
    if not text:
        return sig
    subject = text.splitlines()[0]
    match = None
    for brand, pattern in sig_db.DETECT_SUBJECT_PREFIXES:
        m = pattern.search(subject)
        if m:
            match = (brand, m.group(0))
            break
    if match is None:
        for brand, pattern in sig_db.DETECT_BODY_MARKERS:
            m = pattern.search(text)
            if m:
                match = (brand, m.group(0))
                break
    if match:
        sig.message_marker_match = True
        sig.agent_brand, sig.message_marker_text = match
        sig.label_source = "message_marker"
        sig.label = config.LABEL_AGENT
    return sig


def detect_bot_account(author_login: Optional[str]) -> LabelSignal:
    """Look up a GitHub account login against the known bot-account map."""
    sig = LabelSignal()
    if not author_login:
        return sig
    login = author_login.strip()
    brand = sig_db.brand_for_login(login)
    if brand:
        sig.is_bot_account = True
        sig.bot_account_login = login
        sig.agent_brand = brand
        sig.label_source = "bot_account"
        sig.label = config.LABEL_AGENT
    elif sig_db.is_ci_bot(login) or login.lower().endswith("[bot]"):
        # a recognised bot, but not a coding agent (CI, dependency bumps)
        sig.is_bot_account = True
        sig.bot_account_login = login
    return sig


def detect_email_domain(email: Optional[str], name: Optional[str] = None) -> LabelSignal:
    """Look up an author/committer identity: exact agent addresses, GitHub
    noreply addresses whose login is a known agent account, name+address
    pairs, and aider's `Name (aider)` suffix (source "email_domain",
    i.e. the git identity)."""
    sig = LabelSignal()
    brand = sig_db.brand_for_email(email, name) if email else None
    if brand is None and name and sig_db.AIDER_NAME_RE.search(name):
        brand = sig_db.keep("aider")
    if brand:
        sig.email_domain_match = True
        sig.email_domain = (email or name or "").strip().lower()
        sig.agent_brand = brand
        sig.label_source = "email_domain"
        sig.label = config.LABEL_AGENT
    return sig


def detect_pr_marker(title: str = "", body: str = "",
                     branch_name: str = "", labels: Optional[Iterable[str]] = None) -> LabelSignal:
    """Look for an agent marker on a PR: an agent branch prefix, an exact
    agent label, or an agent-written body marker. The title is accepted
    for API compatibility but brand words in it are not evidence."""
    sig = LabelSignal()
    found = None
    brand = sig_db.brand_for_branch(branch_name)
    if brand:
        found = (brand, f"branch:{branch_name}")
    if found is None:
        for lbl in labels or []:
            brand = sig_db.brand_for_label(lbl)
            if brand:
                found = (brand, f"label:{lbl}")
                break
    if found is None and body:
        marker = detect_message_marker(body)
        if marker.label == config.LABEL_AGENT:
            found = (marker.agent_brand, f"body:{marker.message_marker_text}")
    if found:
        sig.pr_marker_match = True
        sig.agent_brand, sig.pr_marker_text = found[0], found[1][:200]
        sig.label_source = "pr_marker"
        sig.label = config.LABEL_AGENT
    return sig


def resolve_label(
    *,
    commit_message: str = "",
    author_login: Optional[str] = None,
    author_email: Optional[str] = None,
    committer_email: Optional[str] = None,
    pr_title: str = "",
    pr_body: str = "",
    pr_branch_name: str = "",
    pr_labels: Optional[Iterable[str]] = None,
    author_name: Optional[str] = None,
    committer_name: Optional[str] = None,
) -> LabelSignal:
    """Combine every available signal and resolve a single LabelSignal,
    respecting config.LABEL_SOURCE_PRIORITY for which brand/source gets
    reported, while keeping every raw sub-signal that fired (and every
    brand any of them named) on the result.
    """
    trailer_sig = detect_trailer(commit_message)
    marker_sig = detect_message_marker(commit_message)
    bot_sig = detect_bot_account(author_login)
    email_sig = detect_email_domain(author_email, author_name) if (author_email or author_name) else LabelSignal()
    if not email_sig.email_domain_match and (committer_email or committer_name):
        committer_sig = detect_email_domain(committer_email, committer_name)
        if committer_sig.email_domain_match:
            email_sig = committer_sig
    pr_sig = detect_pr_marker(pr_title, pr_body, pr_branch_name, pr_labels)

    by_source = {
        "trailer": trailer_sig,
        "message_marker": marker_sig,
        "bot_account": bot_sig,
        "email_domain": email_sig,
        "pr_marker": pr_sig,
    }

    result = LabelSignal(
        has_trailer=trailer_sig.has_trailer,
        trailer_text=trailer_sig.trailer_text,
        message_marker_match=marker_sig.message_marker_match,
        message_marker_text=marker_sig.message_marker_text,
        is_bot_account=bot_sig.is_bot_account,
        bot_account_login=bot_sig.bot_account_login,
        email_domain_match=email_sig.email_domain_match,
        email_domain=email_sig.email_domain,
        pr_marker_match=pr_sig.pr_marker_match,
        pr_marker_text=pr_sig.pr_marker_text,
    )
    brands: List[str] = []
    for source in config.LABEL_SOURCE_PRIORITY:
        s = by_source[source]
        if s.label == config.LABEL_AGENT and s.agent_brand not in brands:
            brands.append(s.agent_brand)
    result.brands = brands

    for source in config.LABEL_SOURCE_PRIORITY:
        s = by_source[source]
        if s.label == config.LABEL_AGENT:
            result.agent_brand = s.agent_brand
            result.label_source = source
            result.label = config.LABEL_AGENT
            return result

    result.label = config.LABEL_UNLABELED
    result.agent_brand = None
    result.label_source = None
    return result


def apply_pr_label(target: LabelSignal, pr_label: LabelSignal, pr_number: int) -> None:
    """Propagate a PR's agent label onto the commit GitHub created when
    merging it (squash or merge commit), as the lowest-priority source.
    Never overrides a stronger commit-level signal."""
    if pr_label.label != config.LABEL_AGENT:
        return
    if pr_label.agent_brand not in target.brands:
        target.brands.append(pr_label.agent_brand)
    evidence = pr_label.pr_marker_text or pr_label.bot_account_login or pr_label.label_source
    target.pr_marker_match = True
    target.pr_marker_text = f"PR #{pr_number} {evidence}"[:200]
    target.propagated_from_pr = pr_number
    if target.label != config.LABEL_AGENT:
        target.agent_brand = pr_label.agent_brand
        target.label_source = "pr_marker"
        target.label = config.LABEL_AGENT
