"""Repeated game-level shuffled evaluation: how stable is the MLP
ensemble's advantage over the logistic (stand x strikes) baseline
across many different random train/dev partitions of all 164 games?

Architecture and feature set are frozen -- every split refits the same
logistic model and the same frozen 10-seed MLP config
(pitch_sitch.mlp.SELECTED_CONFIG). No model or feature changes are made
in response to any individual split's results; this is purely a
stability read, not a tuning loop. The original 131/33 split (seed=0)
is included as split #1 for reference -- it should reproduce the
already-documented historical numbers exactly.

This is not a substitute for eventually testing once on new, real
future Gausman games -- it only checks stability across resamples of
the currently available 164 games.

Usage:
    PYTHONPATH=. python3 scripts/run_split_stability_check.py --n-repeats 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import build_batter_hand_onehot, build_stand_strikes_interaction, fit_location_means
from pitch_sitch.evaluation import brier_score
from pitch_sitch.mlp import CLASSES, SEEDS, SELECTED_CONFIG, build_features, fit_seed_ensemble
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, load_clean_pitch_log, numeric_columns_for
from pitch_sitch.split import assign_game_split


def average_probabilities(proba_dict: dict, classes=CLASSES) -> pd.DataFrame:
    stacked = np.stack([p[classes].to_numpy() for p in proba_dict.values()], axis=0)
    return pd.DataFrame(stacked.mean(axis=0), columns=classes)


def run_one_split(df: pd.DataFrame, seed: int, test_frac: float, numeric_cols: list[str]) -> dict:
    split_df = assign_game_split(df, test_frac=test_frac, seed=seed)
    train = split_df[split_df["split"] == "train"].reset_index(drop=True)
    dev = split_df[split_df["split"] == "test"].reset_index(drop=True)

    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    X_train = pd.concat(
        [build_features(train, location_means)], axis=1
    )  # build_features already includes stand + stand x strikes
    X_dev = build_features(dev, location_means)
    X_train_s, X_dev_s = scale_numeric(X_train, X_dev, numeric_cols)
    y_train, y_dev = train["pitch_class"], dev["pitch_class"]

    lr = fit_logistic(X_train_s, y_train, class_weight=None)
    proba_lr = predict_proba_df(lr, X_dev_s, CLASSES)

    proba_by_seed = fit_seed_ensemble(
        X_train_s, y_train, X_dev_s,
        SELECTED_CONFIG["hidden_layer_sizes"], SELECTED_CONFIG["alpha"], SELECTED_CONFIG["best_epoch"],
        seeds=SEEDS,
    )
    proba_ens = average_probabilities(proba_by_seed)

    m_lr = evaluate(y_dev, proba_lr, CLASSES)
    m_ens = evaluate(y_dev, proba_ens, CLASSES)

    return {
        "seed": seed,
        "n_train_games": int(train["game_pk"].nunique()),
        "n_dev_games": int(dev["game_pk"].nunique()),
        "n_dev_pitches": len(dev),
        "logistic_accuracy": m_lr["accuracy"],
        "logistic_log_loss": m_lr["log_loss"],
        "logistic_brier": brier_score(y_dev, proba_lr, CLASSES),
        "ensemble_accuracy": m_ens["accuracy"],
        "ensemble_log_loss": m_ens["log_loss"],
        "ensemble_brier": brier_score(y_dev, proba_ens, CLASSES),
        "acc_diff": m_ens["accuracy"] - m_lr["accuracy"],
        "log_loss_diff": m_ens["log_loss"] - m_lr["log_loss"],
        "brier_diff": brier_score(y_dev, proba_ens, CLASSES) - brier_score(y_dev, proba_lr, CLASSES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated game-level shuffled stability check.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--score-diff-bound", type=int, default=6)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("data/processed/split_stability_results.csv"))
    args = parser.parse_args()

    df = load_clean_pitch_log(args.cache_file, args.score_diff_bound)
    numeric_cols = numeric_columns_for(RICHEST_FLAGS)
    print(f"Full cleaned pool: {len(df)} pitches, {df['game_pk'].nunique()} games\n")
    print(f"Running {args.n_repeats} random game-level splits (seeds 0..{args.n_repeats - 1}, test_frac={args.test_frac}).")
    print("seed=0 reproduces the original historical 131/33 split exactly -- shown first as a sanity check.\n")

    results = []
    for seed in range(args.n_repeats):
        r = run_one_split(df, seed, args.test_frac, numeric_cols)
        results.append(r)
        tag = "  <- historical split" if seed == 0 else ""
        print(
            f"  split seed={seed:<3} ({r['n_train_games']}/{r['n_dev_games']} games): "
            f"logistic acc={r['logistic_accuracy']:.4f} ll={r['logistic_log_loss']:.4f} | "
            f"ensemble acc={r['ensemble_accuracy']:.4f} ll={r['ensemble_log_loss']:.4f} | "
            f"diff acc={r['acc_diff']:+.4f} ll={r['log_loss_diff']:+.4f}{tag}"
        )

    res_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out, index=False)

    print(f"\nSaved all {len(res_df)} splits' results to {args.out}\n")
    print("=" * 90)
    print("Summary across all splits:")
    for col in ["logistic_accuracy", "logistic_log_loss", "ensemble_accuracy", "ensemble_log_loss",
                "acc_diff", "log_loss_diff", "brier_diff"]:
        s = res_df[col]
        print(f"  {col:<20} mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"range=[{s.min():+.4f}, {s.max():+.4f}]")

    print()
    print(f"Ensemble beats logistic on accuracy in {int((res_df['acc_diff'] > 0).sum())}/{len(res_df)} splits.")
    print(f"Ensemble beats logistic on log loss in {int((res_df['log_loss_diff'] < 0).sum())}/{len(res_df)} splits.")
    print(f"Ensemble beats logistic on Brier score in {int((res_df['brier_diff'] < 0).sum())}/{len(res_df)} splits.")


if __name__ == "__main__":
    main()
