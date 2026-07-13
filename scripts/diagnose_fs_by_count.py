"""Does the early-count FS underprediction from the game walkthrough hold
across the full 33-game development set?

Uses the current model exactly as-is (richest sequencing, unweighted).
Reports actual FS rate, mean predicted P(FS), FS recall, and calibration
by count, then breaks 0-0 and 1-0 down by batter handedness if sample
sizes permit. Read-only: no model, feature, or split changes.
"""

import argparse
from pathlib import Path

from pitch_sitch.evaluation import class_diagnostic_by_group
from pitch_sitch.models import predict_proba_df
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, fit_sequence_logistic, load_sequence_data

CLASSES = ["FF", "FS", "SL", "OTHER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="FS probability diagnostic by count and handedness.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    model, X_train, X_test = fit_sequence_logistic(train, test, RICHEST_FLAGS, class_weight=None)
    proba = predict_proba_df(model, X_test, CLASSES)
    pred = proba.idxmax(axis=1)

    print(f"Dev/validation set: {len(test)} pitches, {test['game_pk'].nunique()} games\n")

    by_count = class_diagnostic_by_group(test, test["pitch_class"], proba["FS"], pred, ["balls", "strikes"], "FS")
    by_count = by_count.sort_values(["balls", "strikes"])
    print("FS diagnostic by count (all 12 states):")
    print(by_count.to_string(index=False))
    print()

    early = by_count[((by_count["balls"] == 0) & (by_count["strikes"] == 0)) | ((by_count["balls"] == 1) & (by_count["strikes"] == 0))]
    print("Focus: 0-0 and 1-0 --")
    print(early.to_string(index=False))
    print()

    print("0-0 and 1-0 broken down by batter handedness (stand):")
    subset = test[((test["balls"] == 0) & (test["strikes"] == 0)) | ((test["balls"] == 1) & (test["strikes"] == 0))]
    mask = subset.index
    by_count_stand = class_diagnostic_by_group(
        test.loc[mask], test.loc[mask, "pitch_class"], proba.loc[mask, "FS"], pred.loc[mask],
        ["balls", "strikes", "stand"], "FS",
    )
    print(by_count_stand.sort_values(["balls", "strikes", "stand"]).to_string(index=False))


if __name__ == "__main__":
    main()
