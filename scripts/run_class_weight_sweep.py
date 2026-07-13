"""Controlled class-weight sweep, features and split held fixed.

Uses the richest sequencing feature set (+prev3_class) throughout --
the same feature matrix for every point in the sweep -- and varies only
class_weight, interpolating from unweighted (alpha=0) to scikit-learn's
inverse-frequency "balanced" weighting (alpha=1). This isolates the
weighting effect from the feature effect already characterized in
scripts/run_sequence_features.py: nothing here changes what the model
sees, only how much each class's training loss is weighted.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from pitch_sitch.baseline import evaluate
from pitch_sitch.evaluation import confidence_coverage, per_class_report, predicted_class_frequencies
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, fit_sequence_logistic
from pitch_sitch.sequence_pipeline import load_sequence_data
from pitch_sitch.models import predict_proba_df

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = unweighted, 1 = fully "balanced"


def interpolated_weights(balanced: dict, alpha: float) -> dict:
    return {c: 1.0 + alpha * (balanced[c] - 1.0) for c in balanced}


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-weight sweep for the richest sequencing model.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    y_train = train["pitch_class"]
    y_test = test["pitch_class"]
    print(f"Train pitches: {len(train)}  Test pitches: {len(test)}")
    print(f"Feature set held fixed: richest sequencing (+prev3_class)\n")

    balanced_arr = compute_class_weight(class_weight="balanced", classes=np.array(CLASSES), y=y_train)
    balanced = dict(zip(CLASSES, balanced_arr))
    print(f"Train class counts: {y_train.value_counts().reindex(CLASSES).to_dict()}")
    print(f"sklearn 'balanced' weights (alpha=1.0 endpoint): {balanced}\n")

    summary_rows = []

    for alpha in ALPHAS:
        weights = interpolated_weights(balanced, alpha)
        model, X_train, X_test = fit_sequence_logistic(train, test, RICHEST_FLAGS, class_weight=weights)
        proba = predict_proba_df(model, X_test, CLASSES)
        y_pred = proba.idxmax(axis=1)

        metrics = evaluate(y_test, proba, CLASSES)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        pred_freq = predicted_class_frequencies(y_pred, CLASSES)
        per_class = per_class_report(y_test, y_pred, CLASSES)
        cov = confidence_coverage(y_test, proba, CLASSES, THRESHOLDS)

        print("=" * 78)
        print(f"alpha={alpha}  weights={ {c: round(w, 2) for c, w in weights.items()} }")
        print(f"  accuracy={metrics['accuracy']:.4f}  balanced_accuracy={bal_acc:.4f}  log_loss={metrics['log_loss']:.4f}")
        print()
        print("  Per-class precision / recall / F1 / support:")
        print(per_class.to_string())
        print()
        print("  Predicted class frequencies:")
        print(pred_freq.to_string())
        print()
        print("  Confidence vs. coverage (calibration):")
        print(cov.to_string(index=False))
        print()

        summary_rows.append(
            {
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": bal_acc,
                "log_loss": metrics["log_loss"],
                "FF_recall": per_class.loc["FF", "recall"],
                "FS_recall": per_class.loc["FS", "recall"],
                "SL_recall": per_class.loc["SL", "recall"],
                "SL_precision": per_class.loc["SL", "precision"],
                "OTHER_recall": per_class.loc["OTHER", "recall"],
                "SL_predicted_freq": pred_freq["SL"],
                "OTHER_predicted_freq": pred_freq["OTHER"],
            }
        )

    print("=" * 78)
    print("Summary across the sweep:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
