"""Test current-game workload features, one group at a time, against the
handedness baseline (richest sequencing + stand). Each group is an
isolated addition -- not combined with the others -- so its individual
effect isn't confounded with the others. No interactions, no model
changes.

Groups tested: game_pitch_count, times_through_order,
inning_pitch_count, prior_pa_vs_batter. All are leakage-safe by
construction (pitch_sitch.workload_features) -- computed from strictly
prior pitches/plate-appearances in the same game.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import (
    build_batter_hand_onehot,
    build_prior_pa_onehot,
    build_times_through_order_onehot,
    fit_location_means,
)
from pitch_sitch.evaluation import confidence_coverage
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features, load_sequence_data, numeric_columns_for

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]


def build_baseline_matrix(df: pd.DataFrame, location_means: dict) -> pd.DataFrame:
    return pd.concat(
        [build_step_features(df, RICHEST_FLAGS, location_means), build_batter_hand_onehot(df)], axis=1
    )


def run_variant(name, X_train, X_test, y_train, y_test, numeric_cols):
    X_train, X_test = scale_numeric(X_train, X_test, numeric_cols)
    model = fit_logistic(X_train, y_train, class_weight=None)
    proba = predict_proba_df(model, X_test, CLASSES)

    metrics = evaluate(y_test, proba, CLASSES)
    cov = confidence_coverage(y_test, proba, CLASSES, THRESHOLDS)

    print("=" * 78)
    print(f"{name}  (n_features={X_train.shape[1]})")
    print(f"  accuracy={metrics['accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}")
    print("  Confidence vs. coverage (calibration):")
    print(cov.to_string(index=False))
    print()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Workload feature ablation vs. handedness baseline.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    y_train, y_test = train["pitch_class"], test["pitch_class"]

    base_numeric = numeric_columns_for(RICHEST_FLAGS)

    X_train_base = build_baseline_matrix(train, location_means)
    X_test_base = build_baseline_matrix(test, location_means)
    run_variant("Baseline (richest sequencing + stand)", X_train_base.copy(), X_test_base.copy(), y_train, y_test, base_numeric)

    # 1. game_pitch_count (numeric, scaled)
    X_train = X_train_base.copy()
    X_train["game_pitch_count"] = train["game_pitch_count"].reset_index(drop=True)
    X_test = X_test_base.copy()
    X_test["game_pitch_count"] = test["game_pitch_count"].reset_index(drop=True)
    run_variant("+ game_pitch_count", X_train, X_test, y_train, y_test, base_numeric + ["game_pitch_count"])

    # 2. times_through_order (one-hot, 4 categories)
    X_train = pd.concat([X_train_base, build_times_through_order_onehot(train)], axis=1)
    X_test = pd.concat([X_test_base, build_times_through_order_onehot(test)], axis=1)
    run_variant("+ times_through_order", X_train, X_test, y_train, y_test, base_numeric)

    # 3. inning_pitch_count (numeric, scaled)
    X_train = X_train_base.copy()
    X_train["inning_pitch_count"] = train["inning_pitch_count"].reset_index(drop=True)
    X_test = X_test_base.copy()
    X_test["inning_pitch_count"] = test["inning_pitch_count"].reset_index(drop=True)
    run_variant("+ inning_pitch_count", X_train, X_test, y_train, y_test, base_numeric + ["inning_pitch_count"])

    # 4. prior_pa_vs_batter (one-hot, 4 categories)
    X_train = pd.concat([X_train_base, build_prior_pa_onehot(train)], axis=1)
    X_test = pd.concat([X_test_base, build_prior_pa_onehot(test)], axis=1)
    run_variant("+ prior_pa_vs_batter", X_train, X_test, y_train, y_test, base_numeric)


if __name__ == "__main__":
    main()
