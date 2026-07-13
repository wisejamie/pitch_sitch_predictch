"""Inspect the count_game logistic regression on the held-out test set.

Read-only: fits the same model as scripts/run_logistic_baseline.py and
reports it in more detail -- confusion matrix, per-class precision/
recall/F1, balanced accuracy, predicted class frequencies, and
confidence-vs-coverage at several thresholds. Does not change the model.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from pitch_sitch.evaluation import confidence_coverage, confusion, per_class_report, predicted_class_frequencies
from pitch_sitch.models import fit_count_game_logistic, load_clean_split_data

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the count_game logistic regression on the test set.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_clean_split_data(args.cache_file, args.split_file, args.score_diff_bound)
    y_test = test["pitch_class"]

    model, X_train, X_test = fit_count_game_logistic(train, test)
    proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)[CLASSES]
    y_pred = proba.idxmax(axis=1)

    print(f"Test pitches: {len(y_test)}\n")

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion(y_test, y_pred, CLASSES).to_string())
    print()

    print("Per-class precision / recall / F1 / support:")
    print(per_class_report(y_test, y_pred, CLASSES).to_string())
    print()

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print()

    print("Predicted class frequencies (test):")
    print(predicted_class_frequencies(y_pred, CLASSES).to_string())
    print()
    print("Actual class frequencies (test), for comparison:")
    print(y_test.value_counts(normalize=True).reindex(CLASSES).to_string())
    print()

    print(f"Confidence vs. coverage at thresholds {THRESHOLDS}:")
    cov_table = confidence_coverage(y_test, proba, CLASSES, THRESHOLDS)
    print(cov_table.to_string(index=False))


if __name__ == "__main__":
    main()
