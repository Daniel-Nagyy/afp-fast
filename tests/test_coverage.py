"""Tests for the extended coverage: every trace family in
docs/agent-adoption-guide.pdf, repo-level signals, issues, the API-only
mode and the token pool. Example strings are real ones seen on GitHub
(October 2026)."""

import json
from unittest.mock import MagicMock

import pytest

from afp import config, repo_scan
from afp.adoption import summarize_repo
from afp.api_collect import commit_from_node
from afp.features import strip_label_signals
from afp.github_client import GitHubClient
from fake_api import FakeAPI
from afp.heuristics import resolve_label
from afp.issue_collect import record_from_node
from afp.schema import CommitRecord, FileMarker, IssueRecord, RepoSignal
from afp.signatures import BRAND_KIND, brand_for_branch, mentioned_brands

AGENT, UNLABELED = config.LABEL_AGENT, config.LABEL_UNLABELED


@pytest.mark.usefixtures("all_agents")
# --- commit traces ------------------------------------------------------------

@pytest.mark.parametrize("message,brand", [
    ("Fix\n\nAgent-Logs-Url: https://github.com/cloudposse/atmos/sessions/0239174f-a152", "copilot"),
    ("Fix\n\nClaude-Session: https://claude.ai/code/session_014YCqX3r4vyDNPbRy4qi6YT", "claude_code"),
    ("Fix\n\nAmp-Thread-ID: https://ampcode.com/threads/T-01a1705a-0fca", "amp"),
    ("Update page\n\nReplit-Commit-Author: Agent\nReplit-Commit-Session-Id: 5f2c", "replit"),
    ("Redesign UI\n\nX-Lovable-Edit-ID: edt-14ede1d0-d08e", "lovable"),
    ("Fix\n\nMade-with: Cursor", "cursor"),
    ("Fix\n\nAssisted-by: Claude Code:claude-opus-4-8", "claude_code"),
    ("Fix\n\nAssisted-by: Z.ai: GLM 5.3 Flash via Crush <crush@charm.land>", "crush"),
    ("Fix\n\nCo-authored-by: Antigravity <noreply@google.com>", "antigravity"),
    ("Fix\n\nCo-authored-by: Goose <opensource@block.xyz>", "goose"),
    ("Fix\n\nCo-authored-by: Kimi Code <noreply@anthropic.com>", "kimi"),
    ("Fix\n\nCo-authored-by: Replit Agent <agent@replit.com>", "replit"),
    ("Fix\n\nCo-authored-by: v0 <it+v0agent@vercel.com>", "v0"),
    ("Fix\n\nCo-authored-by: Grok 4.7 <261865991+grok-by-xai[bot]@users.noreply.github.com>", "grok"),
    ("Fix\n\nCo-authored-by: Junie <junie@jetbrains.com>", "junie"),
    ("Fix\n\nCo-Authored-By: Warp <agent@warp.dev>", "warp"),
    ("Fix\n\nCo-authored-by: TRAE CLI <traecli@bytedance.com>", "trae"),
    ("Fix\n\nCo-authored-by: Ona <no-reply@ona.com>", "ona"),
    ("Fix\n\nCo-authored-by: Pi <noreply@pi.dev>", "pi"),
    ("Fix\n\nCo-authored-by: CharlieHelps <charlie@charlielabs.ai>", "charlie"),
    ("Fix\n\nCo-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>", "coderabbit"),
    ("Fix\n\n\U0001F916 Generated with [Gemini CLI](https://github.com/google-gemini/gemini-cli)", "gemini"),
    ("Fix\n\n\U0001F916 Generated with [Antigravity CLI](https://antigravity.google)", "antigravity"),
    ("Fix\n\n\U0001F916 Generated with [Qwen Code](https://github.com/QwenLM/qwen-code)", "qwen_coder"),
    ("Fix\n\n\U0001F916 Generated with [Cline](https://cline.bot)", "cline"),
    ("Fix\n\n\U0001F498 Generated with Crush", "crush"),
])
def test_new_commit_traces(message, brand):
    sig = resolve_label(commit_message=message)
    assert sig.label == AGENT and sig.agent_brand == brand


@pytest.mark.usefixtures("all_agents")
@pytest.mark.parametrize("message", [
    "Update page\n\nReplit-Commit-Author: User\nReplit-Commit-Session-Id: 5f2c",   # a human on Replit
    "Fix\n\nCo-authored-by: Jane Doe <noreply@google.com>",                       # google.com alone is not Antigravity
    "Fix\n\nCo-authored-by: Alice <opensource@block.xyz>",
    "Fix\n\nMade-with: love",
    "See the Agent-Logs-Url header docs",
])
def test_new_traces_do_not_overmatch(message):
    assert resolve_label(commit_message=message).label == UNLABELED


@pytest.mark.usefixtures("all_agents")
def test_aider_name_suffix_and_api_author_login():
    sig = resolve_label(commit_message="Add retry", author_name="Jane Doe (aider)", author_email="jane@example.com")
    assert sig.agent_brand == "aider" and sig.label_source == "email_domain"
    sig = resolve_label(commit_message="Build UI", author_login="gpt-engineer-app[bot]")
    assert sig.agent_brand == "lovable" and sig.label_source == "bot_account"


@pytest.mark.usefixtures("all_agents")
def test_tool_trailer_lines_are_stripped_from_features():
    msg = "Update page\n\nBody.\n\nReplit-Commit-Author: Agent\nReplit-Commit-Session-Id: 5f2c\nX-Lovable-Edit-ID: edt-1"
    assert strip_label_signals(msg) == "Update page\n\nBody."


@pytest.mark.usefixtures("all_agents")
# --- PR / issue traces --------------------------------------------------------

@pytest.mark.parametrize("branch,brand", [
    ("cline/tqbcfqec", "cline"), ("Q-DEV-issue-36-1792595619", "amazon_q"),
    ("jules-1689943540043399992-9180178d", "jules"), ("codegen-bot/transcribe-pdf-2703", "codegen"),
    ("open-swe/fix-x", "open_swe"),
])
def test_new_branch_patterns(branch, brand):
    assert brand_for_branch(branch) == brand


@pytest.mark.usefixtures("all_agents")
@pytest.mark.parametrize("text,brands", [
    ("@claude please fix the failing test", ["claude_code"]),
    ("cc @codex review and @copilot", ["codex", "copilot"]),
    ("mail me at me@claude.ai", []),
    ("we use `@claude` in docs", []),
    ("thanks @copilot-pull-request-reviewer", []),
    ("@devin-ai-integration run it", ["devin"]),
])
def test_mentions(text, brands):
    assert mentioned_brands(text) == brands


@pytest.mark.usefixtures("all_agents")
def test_issue_record_from_graphql_node():
    node = {"number": 12, "createdAt": "2026-03-02T10:00:00Z", "body": "@claude can you fix this?",
            "author": {"login": "alice", "__typename": "User"},
            "assignedActors": {"nodes": [{"__typename": "Bot", "login": "Copilot"}]},
            "comments": {"nodes": [{"body": "On it!", "author": {"login": "claude", "__typename": "Bot"}},
                                   {"body": "@jules too", "author": {"login": "bob", "__typename": "User"}}]}}
    rec = record_from_node("o", "r", node)
    assert rec.assignee_brands == ["copilot"]
    assert rec.mention_brands == ["claude_code", "jules"]
    assert rec.agent_commenter_brands == ["claude_code"]


# --- repo signals -------------------------------------------------------------

FILES = {
    ".github/workflows/claude.yml": "jobs:\n  claude:\n    steps:\n      - uses: anthropics/claude-code-action@v1\n",
    ".github/workflows/ci.yml": "jobs:\n  test:\n    steps:\n      - run: npm test\n",
    ".github/workflows/copilot-setup-steps.yml": "jobs: {}\n",
    ".mcp.json": "{\"mcpServers\": {}}",
    "package.json": json.dumps({"devDependencies": {"@openai/codex": "^1.0.0", "react": "18"}}),
    ".devcontainer/devcontainer.json": '{"features": {"ghcr.io/anthropics/devcontainer-features/claude-code:1": {}}}',
    ".gitignore": "node_modules\nCLAUDE.md\n.cursor/\n",
    ".claude/settings.json": json.dumps({"includeCoAuthoredBy": False}),
    ".claude/agents/reviewer.md": "...",
    ".claude/agents/planner.md": "...",
    "AGENTS.md": "Never add Co-authored-by lines to commits.",
}


def _signals(files=FILES):
    paths = list(files)
    assert set(repo_scan.wanted_paths(paths)) <= set(paths)
    return {(s.kind, s.brand): s for s in repo_scan.scan(paths, files.get)}


@pytest.mark.usefixtures("all_agents")
def test_repo_scan_finds_every_kind():
    sig = _signals()
    assert sig[("workflow", "claude_code")].strength == "strong"
    assert ("workflow", "copilot") in sig
    assert ("mcp", "claude_code") in sig
    assert sig[("dependency", "codex")].strength == "weak"
    assert ("devcontainer", "claude_code") in sig
    assert ("gitignore", "claude_code") in sig and ("gitignore", "cursor") in sig
    assert sig[("hidden_attribution", "claude_code")].detail == "commit attribution disabled"
    assert ("hidden_attribution", "unknown") in sig          # AGENTS.md says never add co-authors
    assert sig[("agent_dir", "claude_code")].detail == "2 file(s)"
    assert not any(k == "workflow" and s.path.endswith("ci.yml") for (k, _), s in sig.items())


@pytest.mark.usefixtures("all_agents")
def test_template_gitignore_is_weak():
    files = {".gitignore": "\n".join(["CLAUDE.md", ".cursor/", ".aider*", "GEMINI.md", ".windsurf/", ".roo/"])}
    sig = repo_scan.scan(list(files), files.get)
    assert sig and all(s.strength == "weak" for s in sig)


@pytest.mark.usefixtures("all_agents")
def test_adoption_combines_channels():
    signals = [RepoSignal("workflow", "claude_code", ".github/workflows/claude.yml", "action", "strong",
                          first_added_at="2025-09-01T00:00:00+00:00"),
               RepoSignal("dependency", "codex", "package.json", "@openai/codex", "weak"),
               RepoSignal("hidden_attribution", "claude_code", ".claude/settings.json", "off", "strong")]
    issues = [IssueRecord("o/r", 1, assignee_brands=["copilot"], mention_brands=["jules"])]
    markers = [FileMarker("GEMINI.md", "GEMINI.md", "gemini", first_added_at="2025-10-01T00:00:00+00:00")]
    ad = summarize_repo("o/r", [], [], markers, signals=signals, issues=issues)
    assert set(ad.brands) == {"claude_code", "gemini", "copilot"}   # weak dependency and lone mention left out
    assert ad.signal_brands["dependency"] == ["codex"]
    assert ad.issue_brands == {"assigned": ["copilot"], "mentioned": ["jules"]}
    assert ad.hidden_use is True
    assert (ad.adoption_date, ad.adoption_date_source) == ("2025-09-01T00:00:00+00:00", "workflow")
    assert ad.authoring_brands == []


# --- API-only mode --------------------------------------------------------------

def test_commit_from_graphql_node_uses_login_and_stats():
    node = {"oid": "abc", "message": "Add tests", "authoredDate": "2026-03-01T10:00:00Z",
            "committedDate": "2026-03-01T10:00:00Z", "additions": 40, "deletions": 2, "changedFilesIfAvailable": 3,
            "parents": {"nodes": [{"oid": "p1"}, {"oid": "p2"}], "totalCount": 2},
            "author": {"name": "Copilot", "email": "198982749+Copilot@users.noreply.github.com",
                       "user": {"login": "Copilot"}},
            "committer": {"name": "GitHub", "email": "noreply@github.com"}}
    c = commit_from_node("o/r", node)
    assert (c.additions, c.deletions, c.files_changed, c.is_merge) == (40, 2, 3, True)
    assert c.label.agent_brand == "copilot" and c.label.label_source == "bot_account"


def test_run_pipeline_end_to_end():
    from afp.pipeline import run_pipeline
    result = run_pipeline("o", "r", client=FakeAPI(), since="2026-01-01", as_of="2026-06-30")
    assert result.mode == "api" and result.status == "ok"
    assert result.adoption.commit_brand_counts == {"claude_code": 1}
    assert result.adoption.file_marker_brands == ["claude_code"]
    assert result.adoption.signal_brands == {"workflow": ["codex"]}
    # LOC estimated from file sizes (320,060 code bytes / 32): passes, but flagged as not exactly checked
    assert result.repo_metadata.loc_estimated and result.repo_metadata.total_loc == 10002
    assert result.eligibility.eligible is None and result.eligibility.unchecked
    assert result.repo_metadata.contributor_count == 7
    assert result.feature_vectors[0].edit_velocity is None   # first commit of its author


def test_ineligible_repo_stops_after_the_check():
    from afp.pipeline import run_pipeline

    class Small(FakeAPI):
        def get_tree(self, owner, repo, ref):
            return [{"path": "a.py", "type": "blob", "size": 3_000}], False   # ~94 lines

        def graphql(self, query, variables=None):
            assert "stargazerCount" in query, "only the metadata query runs for an ineligible repo"
            return super().graphql(query, variables)

    small = run_pipeline("o", "r", client=Small(), since="2026-01-01", as_of="2026-06-30")
    assert small.status == "ineligible" and small.commits == [] and small.adoption is None
    assert any("total_loc" in r for r in small.eligibility.reasons)


def test_only_top_agents_are_detected():
    """Other agents' traces are dropped, and a tool that reuses another
    agent's address is not credited to that agent."""
    from afp import signatures as sig_db
    assert sig_db.ACTIVE_BRANDS == frozenset(sig_db.TOP_BRANDS)
    assert resolve_label(commit_message="x\n\nCo-authored-by: Claude <noreply@anthropic.com>").agent_brand \
        == "claude_code"
    assert resolve_label(commit_message="x\n\nCo-authored-by: Kimi Code <noreply@anthropic.com>").label == UNLABELED
    assert resolve_label(commit_message="aider: fix", author_name="Ann (aider)").label == UNLABELED
    assert resolve_label(author_login="google-labs-jules[bot]").agent_brand == "jules"
    assert sig_db.file_marker_brand(".windsurfrules") is None
    assert sig_db.file_marker_brand("CLAUDE.md") == ("CLAUDE.md", "claude_code")
    assert mentioned_brands("@devin @claude please") == ["claude_code"]
    assert ".aider.conf.yml" not in repo_scan.candidate_files()
    assert "requirements.txt" not in repo_scan.candidate_files()
    texts = {".github/workflows/a.yml": "uses: anthropics/claude-code-action@v1\nrun: pip install aider-chat\n",
             "CLAUDE.md": "Never add Co-Authored-By lines to commits.",
             ".windsurfrules": "x"}
    sigs = {(s.kind, s.brand) for s in repo_scan.scan(list(texts), texts.get)}
    assert sigs == {("workflow", "claude_code"), ("hidden_attribution", "claude_code")}
    with pytest.raises(ValueError):
        sig_db.set_active_brands(["claude_code", "nope"])
    assert sig_db.ACTIVE_BRANDS == frozenset(sig_db.TOP_BRANDS)


@pytest.mark.usefixtures("all_agents")
def test_full_tables_resolve_shared_addresses():
    assert resolve_label(commit_message="x\n\nCo-authored-by: Kimi Code <noreply@anthropic.com>").agent_brand == "kimi"


def test_api_commit_keeps_author_timezone_and_raw_message():
    """Regression (clone vs API on 209 repos): API dates were UTC and the
    message lost its blank line."""
    node = {"oid": "x", "message": "Add b\n\nBody line.\n", "authoredDate": "2026-03-01T04:30:00Z",
            "committedDate": "2026-03-01T04:30:00Z", "additions": 2, "deletions": 0, "changedFilesIfAvailable": 1,
            "parents": {"nodes": [{"oid": "p"}], "totalCount": 1},
            "author": {"name": "Dev", "email": "dev@example.com", "date": "2026-03-01T10:00:00+05:30", "user": None},
            "committer": {"name": "Dev", "email": "dev@example.com", "date": "2026-03-01T10:00:00+05:30"}}
    api = commit_from_node("o/r", node)
    assert api.message == "Add b\n\nBody line." and api.author_date == "2026-03-01T10:00:00+05:30"


# --- token pool -------------------------------------------------------------------

def _resp(status=200, json_data=None, headers=None, text=""):
    r = MagicMock()
    r.status_code, r.headers, r.text = status, headers or {}, text
    r.json.return_value = json_data if json_data is not None else {}
    return r


def test_token_pool_switches_token_instead_of_sleeping():
    sleeps = []
    client = GitHubClient(tokens=["tok-a", "tok-b"], sleep=sleeps.append)
    calls = []

    def fake_get(url, timeout=None, headers=None, **kw):
        calls.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer tok-a":
            return _resp(403, text="secondary rate limit", headers={"Retry-After": "600"})
        return _resp(200, {"full_name": "o/r"})
    client.session.get = fake_get
    assert client.get_repo("o", "r")["full_name"] == "o/r"
    assert calls == ["Bearer tok-a", "Bearer tok-b"] and sleeps == []
    client.get_repo("o", "r")                       # tok-a stays parked
    assert calls[-1] == "Bearer tok-b"


def test_rate_limit_waits_do_not_use_up_retries():
    """Regression (500-repo run): a repo failed after five 5-minute
    secondary-rate-limit waits. Waits are now separate from failures."""
    client = GitHubClient(token="t", sleep=lambda s: None)
    limited = _resp(403, text="You have exceeded a secondary rate limit", headers={"Retry-After": "300"})
    client.session.get = MagicMock(side_effect=[limited] * 7 + [_resp(200, {"full_name": "o/r"})])
    assert client.get_repo("o", "r")["full_name"] == "o/r"


def test_dropped_connections_get_a_longer_backoff():
    """Regression (benchmark runs): waves of TLS EOF errors failed whole
    batches after five quick retries; connection errors now get eight,
    with jittered backoff capped at a minute, while 5xx keep five."""
    import requests
    sleeps = []
    client = GitHubClient(token="t", sleep=sleeps.append)
    eof = requests.exceptions.SSLError("EOF occurred in violation of protocol")
    client.session.get = MagicMock(side_effect=[eof] * 7 + [_resp(200, {"full_name": "o/r"})])
    assert client.get_repo("o", "r")["full_name"] == "o/r"
    assert len(sleeps) == 7 and max(sleeps) <= 90
    client.session.get = MagicMock(return_value=_resp(502))
    with pytest.raises(Exception, match="502"):
        client.get_repo("o", "r")
    assert client.session.get.call_count == 5


def test_graphql_rate_limit_error_type_is_waited_out():
    """Regression: GitHub also reports type RATE_LIMIT ("API rate limit
    already exceeded"), which used to fail the repo instead of waiting."""
    sleeps = []
    client = GitHubClient(token="t", sleep=sleeps.append)
    limited = _resp(200, {"errors": [{"type": "RATE_LIMIT", "message": "API rate limit already exceeded"}]})
    client.session.post = MagicMock(side_effect=[limited, _resp(200, {"data": {"ok": 1}})])
    assert client.graphql("{ ok }") == {"ok": 1}
    assert sleeps == [60.0]


def test_empty_200_body_is_retried():
    """Regression (500-repo run): GitHub sent an empty 200 for a tree."""
    client = GitHubClient(token="t", sleep=lambda s: None)
    empty = _resp(200)
    empty.content = b""
    good = _resp(200, {"tree": [], "truncated": False})
    good.content = b'{"tree": []}'
    client.session.get = MagicMock(side_effect=[empty, good])
    assert client.get_tree("o", "r", "main") == ([], False)


def test_token_pool_prefers_token_with_most_budget():
    client = GitHubClient(tokens=["a", "b"])
    client._remaining[(0, "core")] = 100
    client._remaining[(1, "core")] = 4000
    assert client._pick("core") == 1


def test_graphql_page_halves_page_size_on_502():
    """Regression: google/ExoPlayer's issue pages timed out (HTTP 502) five
    times in a row at 50 issues per page and failed the repo."""
    from afp.github_client import GitHubAPIError
    client = GitHubClient(token="t")
    sizes = []

    def fake_graphql(query, variables=None):
        sizes.append(variables["n"])
        if variables["n"] > 12:
            raise GitHubAPIError("Exceeded retries fetching https://api.github.com/graphql (HTTP 502)")
        return {"ok": True}
    client.graphql = fake_graphql
    variables = {"n": 50}
    assert client.graphql_page("q", variables) == {"ok": True}
    assert sizes == [50, 25, 12] and variables["n"] == 12   # later pages keep the smaller size
    client.graphql = MagicMock(side_effect=GitHubAPIError("GraphQL -> 401: bad credentials"))
    with pytest.raises(GitHubAPIError, match="401"):
        client.graphql_page("q", {"n": 50})                   # other errors are not retried


def test_get_blobs_parses_aliases_and_skips_binaries():
    client = GitHubClient(token="t")
    client.graphql = MagicMock(return_value={"repository": {
        "f0": {"byteSize": 10, "isBinary": False, "text": "{}"},
        "f1": None,
        "f2": {"byteSize": 10, "isBinary": True, "text": None}}})
    assert client.get_blobs("o", "r", ["a.json", "missing", "logo.png"]) == {"a.json": "{}"}


def test_every_brand_has_a_kind():
    from afp import signatures as s
    used = set(s.AGENT_EMAILS.values()) | set(s.AGENT_LOGINS.values()) | set(s.BRANCH_PREFIXES.values()) \
        | {b for _, b in s.FILE_MARKERS} | {b for b, _ in s.BODY_MARKER_PATTERNS} | {b for _, b in s.TOOL_TRAILERS} \
        | set(s.MENTIONS.values()) | {b for b, _ in s.TRAILER_NAME_PATTERNS}
    assert used <= set(BRAND_KIND), used - set(BRAND_KIND)
