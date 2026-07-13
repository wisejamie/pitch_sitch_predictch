"""Shared data-loading and feature-building pipeline for the pitch-
sequencing experiments, so any script needing the richest sequencing
feature set (or an intermediate step) builds exactly the same columns
as scripts/run_sequence_features.py, rather than re-implementing it.
"""

from pathlib import Path

import pandas as pd

from pitch_sitch.design_matrix import (
    build_count_game_features,
    build_prev_class_onehot,
    build_prev_location_numeric,
    build_prev_result_onehot,
    clip_score_diff,
    fit_location_means,
)
from pitch_sitch.features import add_runner_flags, add_score_diff
from pitch_sitch.labels import REQUIRED_COLS
from pitch_sitch.sequence_features import ORDER_COLS, add_history, assign_raw_labels
from pitch_sitch.workload_features import add_workload_features

RICHEST_FLAGS = {
    "prev1_class": True,
    "prev1_result": True,
    "prev1_location": True,
    "prev2_class": True,
    "prev3_class": True,
}

NUMERIC_BASE = ["inning", "score_diff"]


def load_sequence_data(cache_file: Path, split_file: Path, score_diff_bound: int = 6):
    df = pd.read_parquet(cache_file)
    df = df.sort_values(ORDER_COLS).reset_index(drop=True)
    df = assign_raw_labels(df)
    df = add_history(df, class_depths=(1, 2, 3), result_depths=(1,), location_depths=(1,))
    df = add_workload_features(df)

    df = df[df[REQUIRED_COLS].notna().all(axis=1)].copy()
    df["pitch_class"] = df["pitch_class_raw"]

    df = add_runner_flags(df)
    df = add_score_diff(df)
    df = clip_score_diff(df, bound=score_diff_bound)

    split = pd.read_csv(split_file)
    df = df.merge(split, on="game_pk", how="inner")

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def build_step_features(df: pd.DataFrame, flags: dict, location_means: dict) -> pd.DataFrame:
    parts = [build_count_game_features(df)]
    if flags.get("prev1_class"):
        parts.append(build_prev_class_onehot(df, 1))
    if flags.get("prev1_result"):
        parts.append(build_prev_result_onehot(df, 1))
    if flags.get("prev1_location"):
        parts.append(build_prev_location_numeric(df, 1, location_means))
    if flags.get("prev2_class"):
        parts.append(build_prev_class_onehot(df, 2))
    if flags.get("prev3_class"):
        parts.append(build_prev_class_onehot(df, 3))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)


def numeric_columns_for(flags: dict) -> list[str]:
    cols = list(NUMERIC_BASE)
    if flags.get("prev1_location"):
        cols += ["prev_1_plate_x", "prev_1_plate_z"]
    return cols


def fit_sequence_logistic(train: pd.DataFrame, test: pd.DataFrame, flags: dict, class_weight=None):
    from pitch_sitch.models import fit_logistic, scale_numeric

    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    X_train = build_step_features(train, flags, location_means)
    X_test = build_step_features(test, flags, location_means)

    numeric_cols = numeric_columns_for(flags)
    X_train, X_test = scale_numeric(X_train, X_test, numeric_cols)

    model = fit_logistic(X_train, train["pitch_class"], class_weight=class_weight)
    return model, X_train, X_test
