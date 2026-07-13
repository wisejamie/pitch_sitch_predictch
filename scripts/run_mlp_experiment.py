"""Does a small regularized MLP improve on the stand x strikes logistic
baseline, using the exact same rows, split, features, target classes,
and unweighted cross-entropy objective?

Hyperparameter selection and early stopping use ONLY a game-level split
carved out of the 131 training games (never the 33-game dev/validation
set). Procedure:
  1. Split train games ~80/20 into inner-train / inner-val (fixed seed).
  2. For each of a small (hidden_layer_sizes x alpha) grid, train with
     manual epoch-by-epoch early stopping on inner-val log loss.
  3. Pick the grid point with the best inner-val log loss; note its
     best epoch count.
  4. Refit on the FULL 131 train games for that fixed epoch count (no
     further peeking at any validation set).
  5. Evaluate once on the 33-game dev/validation set.
  6. Repeat step 4-5 across several random seeds to characterize
     initialization variability.

No new features, no class weighting (MLPClassifier has no class_weight
parameter, so this is enforced by the tool itself, not just by choice).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import build_batter_hand_onehot, build_stand_strikes_interaction, fit_location_means
from pitch_sitch.evaluation import (
    binary_discrimination,
    class_calibration,
    confidence_coverage,
)
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.split import assign_game_split
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features, load_sequence_data, numeric_columns_for

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]

GRID = [
    {"hidden_layer_sizes": (16,), "alpha": 1e-3},
    {"hidden_layer_sizes": (16,), "alpha": 1e-2},
    {"hidden_layer_sizes": (32,), "alpha": 1e-3},
    {"hidden_layer_sizes": (32,), "alpha": 1e-2},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-3},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-2},
]

MAX_EPOCHS = 300
PATIENCE = 20


def build_features(df: pd.DataFrame, location_means: dict) -> pd.DataFrame:
    return pd.concat(
        [
            build_step_features(df, RICHEST_FLAGS, location_means),
            build_batter_hand_onehot(df),
            build_stand_strikes_interaction(df),
        ],
        axis=1,
    )


def train_with_early_stopping(X_tr, y_tr, X_val, y_val, hidden_layer_sizes, alpha, seed):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )
    classes = np.array(sorted(y_tr.unique()))
    best_val_loss = np.inf
    best_epoch = 0
    since_improved = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(X_tr, y_tr, classes=classes)
        proba_val = predict_proba_df(model, X_val, CLASSES)
        val_loss = evaluate(y_val, proba_val, CLASSES)["log_loss"]
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1
        if since_improved >= PATIENCE:
            break

    return best_val_loss, best_epoch


def fit_mlp_fixed_epochs(X_tr, y_tr, hidden_layer_sizes, alpha, n_epochs, seed):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )
    classes = np.array(sorted(y_tr.unique()))
    for _ in range(n_epochs):
        model.partial_fit(X_tr, y_tr, classes=classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Small regularized MLP vs. stand x strikes logistic baseline.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    numeric_cols = numeric_columns_for(RICHEST_FLAGS)
    y_train, y_test = train["pitch_class"], test["pitch_class"]

    # ---- Logistic baseline (stand x strikes) ----
    X_train_lr = build_features(train, location_means)
    X_test_lr = build_features(test, location_means)
    X_train_lr_s, X_test_lr_s = scale_numeric(X_train_lr, X_test_lr, numeric_cols)
    lr_model = fit_logistic(X_train_lr_s, y_train, class_weight=None)
    proba_lr = predict_proba_df(lr_model, X_test_lr_s, CLASSES)
    metrics_lr = evaluate(y_test, proba_lr, CLASSES)
    print(f"Logistic baseline (stand x strikes): accuracy={metrics_lr['accuracy']:.4f}  log_loss={metrics_lr['log_loss']:.4f}\n")

    # ---- Inner game-level split of TRAIN only, for MLP model selection ----
    inner = assign_game_split(train, test_frac=0.2, seed=123)
    inner_train = inner[inner["split"] == "train"].reset_index(drop=True)
    inner_val = inner[inner["split"] == "test"].reset_index(drop=True)
    print(f"Inner split for MLP selection: {inner_train['game_pk'].nunique()} inner-train games "
          f"({len(inner_train)} pitches), {inner_val['game_pk'].nunique()} inner-val games ({len(inner_val)} pitches)\n")

    X_inner_train = build_features(inner_train, location_means)
    X_inner_val = build_features(inner_val, location_means)
    X_inner_train_s, X_inner_val_s = scale_numeric(X_inner_train, X_inner_val, numeric_cols)
    y_inner_train, y_inner_val = inner_train["pitch_class"], inner_val["pitch_class"]

    print("Hyperparameter grid search (selecting on inner-val log loss only):")
    results = []
    for cfg in GRID:
        val_loss, best_epoch = train_with_early_stopping(
            X_inner_train_s, y_inner_train, X_inner_val_s, y_inner_val,
            cfg["hidden_layer_sizes"], cfg["alpha"], seed=0,
        )
        print(f"  hidden={cfg['hidden_layer_sizes']!s:<12} alpha={cfg['alpha']:<8} "
              f"inner_val_log_loss={val_loss:.4f}  best_epoch={best_epoch}")
        results.append({**cfg, "val_loss": val_loss, "best_epoch": best_epoch})

    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nSelected: hidden={best['hidden_layer_sizes']}  alpha={best['alpha']}  epochs={best['best_epoch']}\n")

    # ---- Refit on the FULL train set for the selected epoch count, across seeds ----
    seed_metrics = []
    proba_seeds = []
    for seed in range(args.n_seeds):
        model = fit_mlp_fixed_epochs(
            X_train_lr_s, y_train, best["hidden_layer_sizes"], best["alpha"], best["best_epoch"], seed=seed
        )
        proba = predict_proba_df(model, X_test_lr_s, CLASSES)
        m = evaluate(y_test, proba, CLASSES)
        seed_metrics.append(m)
        proba_seeds.append(proba)
        print(f"  seed={seed}  accuracy={m['accuracy']:.4f}  log_loss={m['log_loss']:.4f}")

    accs = np.array([m["accuracy"] for m in seed_metrics])
    lls = np.array([m["log_loss"] for m in seed_metrics])
    print(f"\nAcross {args.n_seeds} seeds: accuracy mean={accs.mean():.4f} std={accs.std():.4f} "
          f"range=[{accs.min():.4f}, {accs.max():.4f}]")
    print(f"                       log_loss mean={lls.mean():.4f} std={lls.std():.4f} "
          f"range=[{lls.min():.4f}, {lls.max():.4f}]\n")

    # Use the median-log-loss seed as "the" MLP for detailed comparison
    median_idx = int(np.argsort(lls)[len(lls) // 2])
    proba_mlp = proba_seeds[median_idx]
    metrics_mlp = seed_metrics[median_idx]
    print(f"Using seed={median_idx} (median log_loss) as representative MLP for detailed comparison.\n")

    print("=" * 90)
    print(f"{'metric':<20}{'logistic (stand x strikes)':>28}{'MLP (representative seed)':>28}")
    print(f"{'accuracy':<20}{metrics_lr['accuracy']:>28.4f}{metrics_mlp['accuracy']:>28.4f}")
    print(f"{'log_loss':<20}{metrics_lr['log_loss']:>28.4f}{metrics_mlp['log_loss']:>28.4f}")
    mean_conf_lr = proba_lr[CLASSES].max(axis=1).mean()
    mean_conf_mlp = proba_mlp[CLASSES].max(axis=1).mean()
    print(f"{'mean top prob':<20}{mean_conf_lr:>28.4f}{mean_conf_mlp:>28.4f}")
    print()

    print("Confidence vs. coverage -- logistic:")
    print(confidence_coverage(y_test, proba_lr, CLASSES, THRESHOLDS).to_string(index=False))
    print("\nConfidence vs. coverage -- MLP:")
    print(confidence_coverage(y_test, proba_mlp, CLASSES, THRESHOLDS).to_string(index=False))
    print()

    print("Per-class discrimination (ROC-AUC / PR-AUC), logistic vs MLP:")
    for c in CLASSES:
        d_lr = binary_discrimination(y_test, proba_lr[c], c)
        d_mlp = binary_discrimination(y_test, proba_mlp[c], c)
        print(f"  {c:<6} logistic: ROC-AUC={d_lr['roc_auc']:.4f} PR-AUC={d_lr['pr_auc']:.4f}   "
              f"MLP: ROC-AUC={d_mlp['roc_auc']:.4f} PR-AUC={d_mlp['pr_auc']:.4f}")
    print()

    print("Per-class calibration (5 bins), logistic vs MLP -- mean |gap| across bins:")
    for c in CLASSES:
        cal_lr = class_calibration(y_test, proba_lr[c], c, n_bins=5)
        cal_mlp = class_calibration(y_test, proba_mlp[c], c, n_bins=5)
        print(f"  {c:<6} logistic mean|gap|={cal_lr['gap'].abs().mean():.4f}   "
              f"MLP mean|gap|={cal_mlp['gap'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
