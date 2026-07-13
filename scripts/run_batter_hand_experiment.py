"""Add batter handedness as the only new feature, on top of the richest
sequencing feature set. Same unweighted logistic regression, same split.

Tests two things: (1) does it move overall accuracy/log_loss, and (2)
does it close the batter-handedness-driven FS probability gap at 0-0
and 1-0 counts found in scripts/diagnose_fs_by_count.py. Does not add
workload features or change model type -- isolated single-feature test.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import build_batter_hand_onehot, fit_location_means
from pitch_sitch.evaluation import class_diagnostic_by_group
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features, load_sequence_data, numeric_columns_for

CLASSES = ["FF", "FS", "SL", "OTHER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated test: add batter handedness only.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])

    X_train = pd.concat(
        [build_step_features(train, RICHEST_FLAGS, location_means), build_batter_hand_onehot(train)], axis=1
    )
    X_test = pd.concat(
        [build_step_features(test, RICHEST_FLAGS, location_means), build_batter_hand_onehot(test)], axis=1
    )

    numeric_cols = numeric_columns_for(RICHEST_FLAGS)
    X_train, X_test = scale_numeric(X_train, X_test, numeric_cols)

    model = fit_logistic(X_train, train["pitch_class"], class_weight=None)
    proba = predict_proba_df(model, X_test, CLASSES)
    pred = proba.idxmax(axis=1)

    metrics = evaluate(test["pitch_class"], proba, CLASSES)
    print(f"n_features: {X_train.shape[1]} (was 48 without stand)")
    print(f"accuracy={metrics['accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}")
    print("(baseline without stand: accuracy=0.5694  log_loss=0.9371)\n")

    subset_mask = ((test["balls"] == 0) & (test["strikes"] == 0)) | ((test["balls"] == 1) & (test["strikes"] == 0))
    idx = test.index[subset_mask]

    print("FS diagnostic by count x handedness, WITH stand feature:")
    by_count_stand = class_diagnostic_by_group(
        test.loc[idx], test.loc[idx, "pitch_class"], proba.loc[idx, "FS"], pred.loc[idx],
        ["balls", "strikes", "stand"], "FS",
    )
    print(by_count_stand.sort_values(["balls", "strikes", "stand"]).to_string(index=False))
    print()

    print("For reference, WITHOUT stand (from scripts/diagnose_fs_by_count.py):")
    print(" balls  strikes stand   n  actual_rate  mean_predicted  calibration_gap   recall")
    print("     0        0     L 378     0.283069        0.196934        -0.086134 0.000000")
    print("     0        0     R 414     0.101449        0.201149         0.099700 0.000000")
    print("     1        0     L 151     0.456954        0.320723        -0.136230 0.130435")
    print("     1        0     R 124     0.217742        0.304260         0.086518 0.111111")


if __name__ == "__main__":
    main()
