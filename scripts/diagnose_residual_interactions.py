"""Do residual FS-probability errors suggest a count x handedness or
count x previous-pitch interaction the additive handedness-baseline
model can't capture? Read-only diagnostic -- no features or model
changes. Uses the handedness baseline (richest sequencing + stand),
since none of the workload additions beat it.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.design_matrix import build_batter_hand_onehot, fit_location_means
from pitch_sitch.evaluation import class_diagnostic_by_group
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features, load_sequence_data, numeric_columns_for

CLASSES = ["FF", "FS", "SL", "OTHER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual interaction diagnostic for the handedness baseline.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])

    X_train = pd.concat([build_step_features(train, RICHEST_FLAGS, location_means), build_batter_hand_onehot(train)], axis=1)
    X_test = pd.concat([build_step_features(test, RICHEST_FLAGS, location_means), build_batter_hand_onehot(test)], axis=1)
    X_train, X_test = scale_numeric(X_train, X_test, numeric_columns_for(RICHEST_FLAGS))

    model = fit_logistic(X_train, train["pitch_class"], class_weight=None)
    proba = predict_proba_df(model, X_test, CLASSES)
    pred = proba.idxmax(axis=1)

    print("=" * 90)
    print("1. Count x handedness: full 12-count FS diagnostic, split by stand")
    print("=" * 90)
    by_count_stand = class_diagnostic_by_group(test, test["pitch_class"], proba["FS"], pred, ["balls", "strikes", "stand"], "FS")
    print(by_count_stand.sort_values(["strikes", "balls", "stand"]).to_string(index=False))
    print()

    print("Mean calibration gap by strikes level x stand (pooled over balls, larger n per cell):")
    test2 = test.copy()
    grp = class_diagnostic_by_group(test2, test2["pitch_class"], proba["FS"], pred, ["strikes", "stand"], "FS")
    print(grp.sort_values(["strikes", "stand"]).to_string(index=False))
    print()

    print("=" * 90)
    print("2. Count x previous pitch: two-strike vs not, crossed with prev_1_pitch_class")
    print("=" * 90)
    test3 = test.copy()
    test3["two_strike"] = (test3["strikes"] == 2).astype(int)
    by_prev = class_diagnostic_by_group(
        test3, test3["pitch_class"], proba["FS"], pred, ["two_strike", "prev_1_pitch_class"], "FS"
    )
    print(by_prev.sort_values(["two_strike", "prev_1_pitch_class"]).to_string(index=False))


if __name__ == "__main__":
    main()
