from afp.collect import compute_edit_velocity
from afp.features import commits_to_feature_vectors


def test_feature_vectors_strip_label_source_fields(synthetic_commits):
    """The whole point of CommitFeatureVector is that it carries no raw
    label-source signal -- only pu_label (derived) survives."""
    compute_edit_velocity(synthetic_commits)
    fvs = commits_to_feature_vectors(synthetic_commits)
    for fv in fvs:
        fv_dict = fv.to_dict()
        for forbidden in ("has_trailer", "trailer_text", "is_bot_account",
                          "bot_account_login", "email_domain_match", "email_domain",
                          "pr_marker_match", "pr_marker_text", "agent_brand", "label_source"):
            assert forbidden not in fv_dict


def test_pu_label_matches_commit_label(synthetic_commits):
    compute_edit_velocity(synthetic_commits)
    fvs = {fv.sha: fv for fv in commits_to_feature_vectors(synthetic_commits)}
    for c in synthetic_commits:
        expected = 1 if c.label.label == "agent_labeled" else 0
        assert fvs[c.sha].pu_label == expected
