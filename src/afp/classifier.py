"""
Baseline classifier training stub, per the Methods "Models and
Evaluation" plan: interpretable models first (logistic regression),
gradient-boosted trees as a stronger comparison, SHAP-ready.

This is a STUB meant to be extended once real corpus-scale data exists --
it trains on whatever features.jsonl is handed to it and prints a basic
report. It is NOT a finished evaluation pipeline: it does not yet do the
repo-level train/test split, the per-brand holdout, or the manual-label
comparison described in the proposal's Methods section. Those need the
actual labeled corpus the group collects.

Important PU-learning caveat: `pu_label` (1 = agent_labeled, 0 =
unlabeled) is NOT a ground-truth human/agent label -- "0" commits include
unmarked agent commits. Treat any reported recall/precision here as a
lower bound on performance against a cleaner, manually labeled set
(see schema.CommitFeatureVector and Proposal "Manually Labeled
Validation Set").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Feature columns actually handed to the model. Deliberately excludes
# repo_full_name, sha, and anything derived from label sources.
NUMERIC_FEATURES = [
    "message_length", "message_line_count", "template_likeness",
    "seconds_since_prev_commit", "hour_of_day", "day_of_week",
    "additions", "deletions", "files_changed", "add_delete_ratio",
    "frac_source", "frac_docs", "frac_config", "frac_test",
    "edit_velocity",
]
BOOLEAN_FEATURES = ["has_conventional_prefix", "is_merge", "edit_velocity_flagged"]


def load_feature_table(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # add_delete_ratio is +inf whenever deletions == 0 (pure-addition commits,
    # which are common). sklearn's imputer/scaler treat inf as invalid input,
    # so it is NaN-ed out here and left for the median imputer to fill, same
    # as any other missing value, rather than crashing the whole pipeline.
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def build_pipeline(model_name: str = "logreg") -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES + BOOLEAN_FEATURES),
    ])

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_name == "gboost":
        clf = GradientBoostingClassifier()
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    return Pipeline([("pre", pre), ("clf", clf)])


def train_from_jsonl(features_jsonl: str, out_path: str,
                      model_name: str = "logreg",
                      test_size: float = 0.25,
                      random_state: int = 42) -> dict:
    df = load_feature_table(features_jsonl)
    if df.empty:
        raise ValueError(f"No rows found in {features_jsonl}")
    missing = [c for c in NUMERIC_FEATURES + BOOLEAN_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"features.jsonl is missing expected columns: {missing}")

    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES]
    y = df["pu_label"]
    groups = df["repo_full_name"]  # repo-level split, per the Methods section

    report = {}
    if groups.nunique() > 1 and y.nunique() > 1:
        splitter = GroupShuffleSplit(test_size=test_size, random_state=random_state, n_splits=1)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_pipeline(model_name)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        print(classification_report(y_test, preds, zero_division=0))
    else:
        pipe = build_pipeline(model_name)
        pipe.fit(X, y)
        report = {"note": "dataset too small/homogeneous for a held-out split; "
                           "model fit on the full set with no evaluation"}
        print(report["note"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return report
