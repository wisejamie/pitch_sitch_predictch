"""Does adding raw prev_1 Statcast bat-tracking measurements (bat speed,
swing length, attack angle/direction, swing path tilt, contact-point
offsets, miss distance) improve on the current working baseline?

These fields are populated only when the previous pitch was swung at
(and not at all before 2023), so this is a genuinely sparse feature
group -- each field carries its own has-value flag and train-mean
imputation (pitch_sitch.design_matrix.build_prev_bat_tracking_numeric),
same pattern as prev_k location.

Lightweight first pass, as instructed:
  - the CURRENT historical 131/33 split (data/processed/game_split.csv),
    not the repeated-shuffle protocol;
  - logistic regression (as always) plus a SINGLE MLP fit at the fixed
    reference seed (pitch_sitch.mlp.REFERENCE_SEED), frozen architecture
    (pitch_sitch.mlp.SELECTED_CONFIG) -- not the full 10-seed ensemble.

The full 30-split stability study (scripts/run_split_stability_check.py)
is deliberately NOT run here -- only escalate to it if this meaningfully
improves accuracy/log_loss over the baseline on this pass.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import build_batter_hand_onehot, build_stand_strikes_interaction, fit_location_means
from pitch_sitch.mlp import CLASSES, REFERENCE_SEED, SELECTED_CONFIG, fit_mlp_fixed_epochs
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import (
    RICHEST_FLAGS,
    bat_tracking_columns_for,
    build_step_features,
    load_sequence_data,
    location_columns_for,
    numeric_columns_for,
)

CANDIDATES = {
    "baseline (current)": dict(RICHEST_FLAGS),
    "+ prev1 bat tracking": {**RICHEST_FLAGS, "prev1_bat_tracking": True},
}


def build_candidate_features(df: pd.DataFrame, flags: dict, means: dict) -> pd.DataFrame:
    return pd.concat(
        [build_step_features(df, flags, means), build_batter_hand_onehot(df), build_stand_strikes_interaction(df)],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test prev_1 raw bat-tracking features (lightweight pass).")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    y_train, y_test = train["pitch_class"], test["pitch_class"]
    print(f"Train: {len(train)} pitches ({train['game_pk'].nunique()} games)  "
          f"Dev: {len(test)} pitches ({test['game_pk'].nunique()} games)\n")

    rows = []
    for name, flags in CANDIDATES.items():
        means = fit_location_means(train, location_columns_for(flags) + bat_tracking_columns_for(flags))
        X_train = build_candidate_features(train, flags, means)
        X_test = build_candidate_features(test, flags, means)
        numeric_cols = numeric_columns_for(flags)
        X_train_s, X_test_s = scale_numeric(X_train, X_test, numeric_cols)

        lr = fit_logistic(X_train_s, y_train, class_weight=None)
        proba_lr = predict_proba_df(lr, X_test_s, CLASSES)
        m_lr = evaluate(y_test, proba_lr, CLASSES)

        mlp = fit_mlp_fixed_epochs(
            X_train_s, y_train,
            SELECTED_CONFIG["hidden_layer_sizes"], SELECTED_CONFIG["alpha"], SELECTED_CONFIG["best_epoch"],
            seed=REFERENCE_SEED,
        )
        proba_mlp = predict_proba_df(mlp, X_test_s, CLASSES)
        m_mlp = evaluate(y_test, proba_mlp, CLASSES)

        print(f"{name}  (n_features={X_train_s.shape[1]})")
        print(f"  logistic:  accuracy={m_lr['accuracy']:.4f}  log_loss={m_lr['log_loss']:.4f}")
        print(f"  MLP(seed={REFERENCE_SEED}): accuracy={m_mlp['accuracy']:.4f}  log_loss={m_mlp['log_loss']:.4f}")
        print()

        rows.append({
            "candidate": name, "n_features": X_train_s.shape[1],
            "logistic_accuracy": m_lr["accuracy"], "logistic_log_loss": m_lr["log_loss"],
            "mlp_accuracy": m_mlp["accuracy"], "mlp_log_loss": m_mlp["log_loss"],
        })

    print("=" * 100)
    print("Summary (deltas vs. baseline row):")
    res = pd.DataFrame(rows).set_index("candidate")
    base = res.loc["baseline (current)"]
    for name in res.index:
        r = res.loc[name]
        print(
            f"  {name:<25} n_feat={int(r['n_features']):<4} "
            f"logistic: acc={r['logistic_accuracy']:.4f} (Δ{r['logistic_accuracy']-base['logistic_accuracy']:+.4f}) "
            f"ll={r['logistic_log_loss']:.4f} (Δ{r['logistic_log_loss']-base['logistic_log_loss']:+.4f})  |  "
            f"MLP: acc={r['mlp_accuracy']:.4f} (Δ{r['mlp_accuracy']-base['mlp_accuracy']:+.4f}) "
            f"ll={r['mlp_log_loss']:.4f} (Δ{r['mlp_log_loss']-base['mlp_log_loss']:+.4f})"
        )


if __name__ == "__main__":
    main()
