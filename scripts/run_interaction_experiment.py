"""Test the count x batter-handedness interaction, three ways, with the
dataset, split, target, unweighted objective, and all other features
held fixed:

  A. additive baseline (richest sequencing + stand)
  B. A + stand x strikes interaction (6 pure interaction columns, added
     on top of -- not replacing -- the existing stand/count terms)
  C. A + stand x full 12-state count interaction (24 pure interaction
     columns, same additive-on-top construction)

Reports accuracy, log loss, confidence/coverage, and FS calibration
within count x handedness groups for all three. FS calibration here
means specifically P(FS): every "calibration_gap" below is
mean_predicted(P(FS)) - actual_rate(pitch_class == FS), via
class_diagnostic_by_group(..., proba["FS"], ..., positive_label="FS").

Finally, treats any improvement as provisional to this development set:
runs grouped (game-level) bootstrap resampling of the *evaluation* set
(models are fit once, not refit per resample) to check whether B's and
C's improvement over A is stable across which games happen to be
included, respecting the game-level dependency in the data. Does not
add other interactions or change model classes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import (
    build_batter_hand_onehot,
    build_stand_count_interaction,
    build_stand_strikes_interaction,
    fit_location_means,
)
from pitch_sitch.evaluation import class_diagnostic_by_group, confidence_coverage
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features, load_sequence_data, numeric_columns_for

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]


def fit_and_predict(X_train, X_test, y_train, numeric_cols):
    X_train, X_test = scale_numeric(X_train, X_test, numeric_cols)
    model = fit_logistic(X_train, y_train, class_weight=None)
    return predict_proba_df(model, X_test, CLASSES), X_train.shape[1]


def report(name, y_test, proba, test_df):
    pred = proba.idxmax(axis=1)
    metrics = evaluate(y_test, proba, CLASSES)
    cov = confidence_coverage(y_test, proba, CLASSES, THRESHOLDS)

    print("=" * 90)
    print(f"Model: {name}")
    print(f"  accuracy={metrics['accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}")
    print("  Confidence vs. coverage (overall calibration):")
    print(cov.to_string(index=False))
    print()

    pooled = class_diagnostic_by_group(test_df, y_test, proba["FS"], pred, ["strikes", "stand"], "FS")
    print("  FS calibration (P(FS) specifically) pooled by strikes x stand:")
    print(pooled.sort_values(["strikes", "stand"]).to_string(index=False))
    print()
    return metrics, pred


def bootstrap_compare(test_df, proba_by_model, y_true, baseline_name, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    games = test_df["game_pk"].unique()
    n_games = len(games)
    game_idx = test_df["game_pk"].to_numpy()
    idx_by_game = {g: np.where(game_idx == g)[0] for g in games}

    other_names = [n for n in proba_by_model if n != baseline_name]
    diffs = {n: {"accuracy": [], "log_loss": []} for n in other_names}

    for _ in range(n_boot):
        sampled_games = rng.choice(games, size=n_games, replace=True)
        idx = np.concatenate([idx_by_game[g] for g in sampled_games])
        y_sample = y_true.iloc[idx].reset_index(drop=True)

        base_m = evaluate(y_sample, proba_by_model[baseline_name].iloc[idx].reset_index(drop=True), CLASSES)
        for n in other_names:
            m = evaluate(y_sample, proba_by_model[n].iloc[idx].reset_index(drop=True), CLASSES)
            diffs[n]["accuracy"].append(m["accuracy"] - base_m["accuracy"])
            diffs[n]["log_loss"].append(m["log_loss"] - base_m["log_loss"])

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Count x handedness interaction test.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    y_train, y_test = train["pitch_class"], test["pitch_class"]
    numeric_cols = numeric_columns_for(RICHEST_FLAGS)

    base_train = pd.concat([build_step_features(train, RICHEST_FLAGS, location_means), build_batter_hand_onehot(train)], axis=1)
    base_test = pd.concat([build_step_features(test, RICHEST_FLAGS, location_means), build_batter_hand_onehot(test)], axis=1)

    # A: additive baseline
    X_train_A, X_test_A = base_train.copy(), base_test.copy()
    proba_A, nfeat_A = fit_and_predict(X_train_A, X_test_A, y_train, numeric_cols)
    metrics_A, pred_A = report(f"A: additive baseline (n_features={nfeat_A})", y_test, proba_A, test)

    # B: + stand x strikes interaction
    X_train_B = pd.concat([base_train, build_stand_strikes_interaction(train)], axis=1)
    X_test_B = pd.concat([base_test, build_stand_strikes_interaction(test)], axis=1)
    proba_B, nfeat_B = fit_and_predict(X_train_B, X_test_B, y_train, numeric_cols)
    metrics_B, pred_B = report(f"B: + stand x strikes interaction (n_features={nfeat_B})", y_test, proba_B, test)

    # C: + stand x full 12-state count interaction
    X_train_C = pd.concat([base_train, build_stand_count_interaction(train)], axis=1)
    X_test_C = pd.concat([base_test, build_stand_count_interaction(test)], axis=1)
    proba_C, nfeat_C = fit_and_predict(X_train_C, X_test_C, y_train, numeric_cols)
    metrics_C, pred_C = report(f"C: + stand x full-count interaction (n_features={nfeat_C})", y_test, proba_C, test)

    print("=" * 90)
    print("Summary:")
    print(f"{'model':<45}{'accuracy':>10}{'log_loss':>10}")
    print(f"{'A: additive baseline':<45}{metrics_A['accuracy']:>10.4f}{metrics_A['log_loss']:>10.4f}")
    print(f"{'B: stand x strikes':<45}{metrics_B['accuracy']:>10.4f}{metrics_B['log_loss']:>10.4f}")
    print(f"{'C: stand x full count':<45}{metrics_C['accuracy']:>10.4f}{metrics_C['log_loss']:>10.4f}")
    print()

    print("Full 12-state x stand FS calibration for model C (most granular interaction):")
    pred_C_series = proba_C.idxmax(axis=1)
    full_table = class_diagnostic_by_group(test, y_test, proba_C["FS"], pred_C_series, ["balls", "strikes", "stand"], "FS")
    print(full_table.sort_values(["strikes", "balls", "stand"]).to_string(index=False))
    print()

    print("=" * 90)
    print(f"Grouped (game-level) bootstrap, n_boot={args.n_boot}, resampling the evaluation set only "
          f"(models fit once, not refit per draw):")
    diffs = bootstrap_compare(
        test, {"A": proba_A, "B": proba_B, "C": proba_C}, y_test, baseline_name="A", n_boot=args.n_boot
    )
    for name in ["B", "C"]:
        acc_diffs = np.array(diffs[name]["accuracy"])
        ll_diffs = np.array(diffs[name]["log_loss"])
        print(f"\n{name} vs A:")
        print(
            f"  accuracy diff:  mean={acc_diffs.mean():+.4f}  95% CI=[{np.percentile(acc_diffs, 2.5):+.4f}, "
            f"{np.percentile(acc_diffs, 97.5):+.4f}]  P(improvement)={np.mean(acc_diffs > 0):.3f}"
        )
        print(
            f"  log_loss diff:  mean={ll_diffs.mean():+.4f}  95% CI=[{np.percentile(ll_diffs, 2.5):+.4f}, "
            f"{np.percentile(ll_diffs, 97.5):+.4f}]  P(improvement)={np.mean(ll_diffs < 0):.3f}"
        )


if __name__ == "__main__":
    main()
