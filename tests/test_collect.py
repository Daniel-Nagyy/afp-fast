from afp import config
from afp.collect import compute_edit_velocity


def test_trailer_commit_labeled_and_sized_correctly(synthetic_commits):
    big = next(c for c in synthetic_commits if c.message.startswith("Implement large feature"))
    assert big.additions == 500
    assert big.deletions == 0
    assert big.label.label == config.LABEL_AGENT
    assert big.label.agent_brand == "claude_code"
    assert big.label.label_source == "trailer"


def test_codex_bot_email_commit_labeled(synthetic_commits):
    gen = next(c for c in synthetic_commits if c.message.startswith("Generate module"))
    assert gen.label.label == config.LABEL_AGENT
    assert gen.label.agent_brand == "codex"
    assert gen.label.label_source == "email_domain"


def test_plain_human_commits_unlabeled(synthetic_commits):
    initial = next(c for c in synthetic_commits if c.message.startswith("Initial commit"))
    assert initial.label.label == config.LABEL_UNLABELED


def test_edit_velocity_flags_fast_large_trailer_commit(synthetic_commits):
    compute_edit_velocity(synthetic_commits)
    big = next(c for c in synthetic_commits if c.message.startswith("Implement large feature"))
    assert big.seconds_since_author_prev_commit == 60.0
    assert big.edit_velocity == 500 / 60.0
    assert big.edit_velocity_flagged is True
    assert big.edit_velocity_excluded_reason is None


def test_edit_velocity_none_for_authors_first_commit(synthetic_commits):
    compute_edit_velocity(synthetic_commits)
    initial = next(c for c in synthetic_commits if c.message.startswith("Initial commit"))
    assert initial.seconds_since_author_prev_commit is None
    assert initial.edit_velocity is None

    # codex-bot has only one commit -> also no prior-commit delta
    gen = next(c for c in synthetic_commits if c.message.startswith("Generate module"))
    assert gen.seconds_since_author_prev_commit is None


def test_edit_velocity_skips_merge_commits(synthetic_commits):
    merge = next(c for c in synthetic_commits if c.message.startswith("chore: update lockfile"))
    merge.is_merge = True
    compute_edit_velocity(synthetic_commits)
    assert merge.edit_velocity_excluded_reason == "merge_commit"
    assert merge.edit_velocity_flagged is False
