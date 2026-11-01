import json
import math

from afp.classifier import load_feature_table, train_from_jsonl


def test_load_feature_table_sanitizes_infinity(tmp_path):
    """Regression test: add_delete_ratio is +inf whenever deletions == 0,
    which is common. This used to crash sklearn's imputer with
    'Input X contains infinity' -- caught during live testing against the
    synthetic repo. load_feature_table must NaN it out instead."""
    jsonl_path = tmp_path / "features.jsonl"
    row = {
        "repo_full_name": "o/r", "sha": "abc",
        "message_length": 10, "message_line_count": 1, "template_likeness": 0.0,
        "seconds_since_prev_commit": 10.0, "hour_of_day": 10, "day_of_week": 1,
        "additions": 5, "deletions": 0, "files_changed": 1,
        "add_delete_ratio": float("inf"),
        "frac_source": 1.0, "frac_docs": 0.0, "frac_config": 0.0, "frac_test": 0.0,
        "is_merge": False, "merge_type": None, "edit_velocity": 0.5,
        "edit_velocity_flagged": False, "has_conventional_prefix": True,
        "pu_label": 0,
    }
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(row) + "\n")

    df = load_feature_table(str(jsonl_path))
    assert math.isnan(df["add_delete_ratio"].iloc[0])


def test_train_from_jsonl_runs_on_tiny_single_repo_dataset(tmp_path):
    """With only one repo and/or one class present, there's no valid
    repo-level split -- train_from_jsonl must fall back to fit-only
    rather than crash (see classifier.py's GroupShuffleSplit branch)."""
    jsonl_path = tmp_path / "features.jsonl"
    rows = []
    for i in range(4):
        rows.append({
            "repo_full_name": "o/r", "sha": f"sha{i}",
            "message_length": 10, "message_line_count": 1, "template_likeness": 0.0,
            "seconds_since_prev_commit": 10.0 * i, "hour_of_day": 10, "day_of_week": 1,
            "additions": 5 * i, "deletions": 0, "files_changed": 1,
            "add_delete_ratio": float("inf"),
            "frac_source": 1.0, "frac_docs": 0.0, "frac_config": 0.0, "frac_test": 0.0,
            "is_merge": False, "merge_type": None, "edit_velocity": 0.1 * i,
            "edit_velocity_flagged": False, "has_conventional_prefix": True,
            "pu_label": i % 2,
        })
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out_path = tmp_path / "model.joblib"
    report = train_from_jsonl(str(jsonl_path), str(out_path))
    assert out_path.exists()
    assert "note" in report
