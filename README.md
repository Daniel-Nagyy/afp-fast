# AFP Fast: top-5 agent adoption, API only

A fast, single-mode version of the Agent Fingerprint Pipeline (AFP) for
the thesis *"Identifying Coding-Agent Fingerprints in Commit Metadata"*.
It measures coding-agent adoption across many GitHub repos using only
the GitHub API (no cloning), checks the methodology's eligibility rules
first, and detects only the five agents that matter most.

There is one way to run it:

1. **Eligibility first.** One GraphQL query (stars, fork/archived, last
   push, full commit count) and the file tree (lines of code estimated
   from file sizes). A repo that fails a rule stops here and is saved with
   status `ineligible`, costing one query and one REST call.
2. **Everything else at once** for eligible repos: commits in the window,
   PRs, issues, contributor count, review rules, and the few small files
   the repo scanner reads, all requested side by side.
3. **Top five agents only**, on every channel: **Claude Code, Copilot,
   Codex, Cursor and Jules**, the five seen most in three 500-repo samples
   (Feb 2025 – Sep 2026: Claude Code 146 repos, Copilot 60, Codex 17,
   Cursor 17, Jules 111 agent commits/PRs). Other agents' traces, generic
   ones (`AGENTS.md`, an `ai-generated` label) and files only other agents
   use are skipped.

Commit labels are always `agent_labeled` or `unlabeled`, **never**
`human`. Features (message, timing, diff size, merge type, edit velocity)
are computed with every label signal stripped first.

## Install

```bash
cd afp-fast
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src                 # PowerShell: $env:PYTHONPATH = 'src'
export GITHUB_TOKEN=github_pat_...     # PowerShell: $env:GITHUB_TOKEN = '...'
```

Needs Python 3.9+ and a GitHub token (a fine-grained token with **no
scopes** is enough for public repos). With several tokens set
`GITHUB_TOKENS=tok1,tok2,tok3`: every request uses the token with the most
budget left, so throughput grows with the number of tokens.

## Use

```bash
# One repo
python -m afp.cli run spf13/viper --out ./out/viper --since 2025-08-12 --until 2026-02-21

# A corpus: any CSV with a full_name column (e.g. a gitcorpus export), or one owner/repo per line
python -m afp.cli batch corpus.csv --out ./out/corpus --since 2025-08-12 --until 2026-02-21 --as-of 2026-02-21

# Pilot on the first 20 repos, then run the rest with the same command
python -m afp.cli batch corpus.csv --out ./out/corpus --since 2025-08-12 --limit 20

# Re-print the corpus summary / rebuild the merged files
python -m afp.cli report ./out/corpus

# Train the baseline classifier on every repo's commits
python -m afp.cli train ./out/corpus/features.jsonl --out model.joblib --model gboost

# Check your token and the API layer
python -m afp.cli smoke-test-api readthedocs/sphinx_rtd_theme
```

| Option | Meaning |
|---|---|
| `--since YYYY-MM-DD` | First day of the window (commits, PRs, issues). |
| `--until YYYY-MM-DD` | Last day of the window. |
| `--as-of YYYY-MM-DD` | Date the "pushed in the last 12 months" rule is measured from (default: today). Use the window's last day. |
| `--max-commits N` | Newest commits per repo (default 2000; `0` = all). |
| `--max-prs N` | Newest PRs per repo (default 300; `0` = all in the window). |
| `--no-prs` | Skip PR collection. |
| `--max-issues N` | Newest issues scanned per repo (default 200; `0` = all). |
| `--no-issues` | Skip issue collection. |

Batch only: `--workers N` (default 24), `--retry-failed`, `--limit N`.
Add `-v` before the command to log retries and rate-limit waits.

## Eligibility rules

A repo is **eligible** when it passes every rule of the shared
methodology: not a fork, not archived, pushed in the 12 months before
`--as-of`, at least 10 stars, at least 5,000 lines of code and at least
100 commits. Lines of code are estimated as code bytes / 32 (on 196 repos
this agreed with exact counts on the 5,000-line rule for 92%), so a repo
that passes everything is reported as `eligible` empty with the LOC rule
under `eligibility_unchecked`, rather than `true`.

## Running on a large corpus

- **Resumable.** Each repo writes `repos/<owner>__<repo>/summary.json`
  last. Re-running the same command skips finished and ineligible repos
  (and failed ones unless `--retry-failed`).
- **One bad repo never stops the run.** Deleted, renamed or private repos
  are recorded as `failed` with the error.
- **Rate limits and dropped connections are handled.** All workers share
  one client that retries 5xx errors, backs off up to about two minutes
  when GitHub drops connections, and waits for rate limits to reset.
- **What limits a big run.** An eligible repo costs about 2 GraphQL points
  and 3–4 REST requests, an ineligible one 1 of each. GitHub allows 5,000
  of each per token per hour, so one token handles roughly 1,200 eligible
  repos an hour (more when many are ineligible). Add tokens to go faster.
- **Measured.** On 150 repos from the Aug 2025 – Feb 2026 sample (one
  token, 24 workers) a run took 29 s and 696 requests, against 53 s and
  1,337 requests for full AFP in API mode, with identical results for the
  56 eligible repos.

## Output

Per repo (`run --out DIR`, or `batch`'s `repos/<owner>__<repo>/`):

| File | Contents |
|---|---|
| `commits.jsonl` | One `CommitRecord` per commit in the window: raw data and label evidence. |
| `prs.jsonl` | One `PullRequestRecord` per PR. |
| `issues.jsonl` | Issues with agent involvement: agent assignees, `@mentions`, agent bot comments. |
| `features.jsonl` | One `CommitFeatureVector` per commit: what feeds the classifier. |
| `file_markers.jsonl` | Agent rules/guidance/config files present at HEAD. |
| `signals.jsonl` | CI workflows, MCP config, dependencies, dev containers, `.gitignore` entries, disabled attribution, agent sub-folders. |
| `summary.json` | Status (`ok`, `ineligible`, `failed`), window, notes, metadata, eligibility, adoption roll-up. |

Ineligible repos have only `summary.json` filled in (metadata and the
failed rules).

Corpus level (`batch --out DIR`, rebuilt by `report`): `repos.csv` (one
row per repo), `adoption_by_brand.csv` (repos per agent and channel, all
and eligible-only), `file_markers.csv`, `signals.csv`, the concatenated
JSONL files (`features.jsonl` is the training set) and
`corpus_summary.json`. CSV files are UTF-8 with a BOM for Excel.

## Detection

Signatures live in `src/afp/signatures.py` (`TOP_BRANDS` is the list of
five), repo-level checks in `src/afp/repo_scan.py`.

| Channel | Examples | Strength |
|---|---|---|
| Commit trailer | `Co-Authored-By: Claude <noreply@anthropic.com>`, `Agent-Logs-Url:`, `Claude-Session:`, `Made-with: Cursor` | strong |
| Commit author | `cursoragent@cursor.com`, `copilot-swe-agent[bot]`, `google-labs-jules[bot]` | strong |
| Message footer | `Generated with [Claude Code]`, `chatgpt.com/codex/tasks/` and `jules.google` links | strong |
| PR | bot author, branch prefix (`claude/`, `codex/`, `copilot/`, `cursor/`, `jules-<n>-<hash>`), label, body footer | strong |
| CI workflow | `anthropics/claude-code-action`, `openai/codex-action`, `cursor-agent`, `copilot-setup-steps.yml` | strong |
| Hidden attribution | `includeCoAuthoredBy: false`; "never add Co-authored-by" in `CLAUDE.md`, `.cursorrules` or `copilot-instructions.md` | strong |
| Issue | Copilot or Jules assigned, agent bot comment | medium |
| Files | `CLAUDE.md`, `.claude/`, `.cursor/`, `.github/copilot-instructions.md`, `.codex/`, `.jules/` | medium |
| MCP / agent folders | `.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`; `.claude/agents/`, `.github/prompts/` | medium |
| Dev container | Claude Code / Codex features, Copilot Chat extension | medium / weak |
| `.gitignore` | `CLAUDE.md`, `.cursor/`, `.codex/` | medium; weak if template-like |
| Dependency | `@anthropic-ai/claude-code`, `@openai/codex`, `@github/copilot` | weak |
| `@mention` | `@claude`, `@codex`, `@copilot`, `@jules` | weak unless the agent also acted |

Every pattern is anchored to where the agent writes it, so "move the
cursor" or a co-author called Claude Martin are not signals. Tools that
reuse one of these agents' addresses (Kimi Code writes as
`noreply@anthropic.com`) are recognised and **not** counted as that agent.
AI review bots reviewing a PR are recorded in `ai_reviewers`, not as
adoption.

**Label-source priority:** trailer > message marker > bot account >
e-mail > PR marker. An agent PR's label is propagated to the squash or
merge commit GitHub created, as the lowest-priority source.

## Limitations

- **No changed paths from the API**, so the file-type-mix features stay 0,
  lockfile-only commits cannot be excluded from edit velocity, and file
  first-added dates are not available.
- **Lines of code are estimated** (see Eligibility rules).
- **Only five agents.** A repo using another agent (Devin, Aider, Kiro, ...)
  counts as having no agent here.
- Detection is a **lower bound**: agents that leave no trace are
  invisible, so `unlabeled` is never treated as human (PU framing).

## Testing

```bash
pytest -q
```

174 tests, all offline: the signature roster (with real true and false
positives, run against the full tables), the top-5 restriction, label
priority and PR propagation, edit velocity, feature stripping, the API
client's retry and rate-limit handling, GraphQL parsing, the pipeline end
to end (eligible and ineligible repos) and batch resume.
