"""Incremental pitch-sequencing feature experiment.

Adds recent-pitch-history features on top of the count_game logistic
regression, one piece at a time: previous pitch class, then previous
result, then previous location, then extending depth to the previous
two and three pitch classes. Split, target classes, model type
(unweighted multinomial logistic regression), and evaluation are all
held fixed from the count_game baseline so results are directly
comparable across steps. Does not introduce class weighting or a new
model -- this is purely a feature-set comparison.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import fit_location_means
from pitch_sitch.evaluation import confidence_coverage, per_class_report, predicted_class_frequencies
from pitch_sitch.sequence_pipeline import build_step_features, load_sequence_data

CLASSES = ["FF", "FS", "SL", "OTHER"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]

STEPS = [
    ("count_game", {}),
    ("+prev1_class", {"prev1_class": True}),
    ("+prev1_result", {"prev1_class": True, "prev1_result": True}),
    ("+prev1_location", {"prev1_class": True, "prev1_result": True, "prev1_location": True}),
    (
        "+prev2_class",
        {"prev1_class": True, "prev1_result": True, "prev1_location": True, "prev2_class": True},
    ),
    (
        "+prev3_class",
        {
            "prev1_class": True,
            "prev1_result": True,
            "prev1_location": True,
            "prev2_class": True,
            "prev3_class": True,
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental pitch-sequencing feature experiment.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    print(f"Train pitches: {len(train)}  Test pitches: {len(test)}\n")

    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])

    y_train = train["pitch_class"]
    y_test = test["pitch_class"]

    summary_rows = []

    for name, flags in STEPS:
        X_train = build_step_features(train, flags, location_means)
        X_test = build_step_features(test, flags, location_means)

        numeric_to_scale = ["inning", "score_diff"]
        if flags.get("prev1_location"):
            numeric_to_scale += ["prev_1_plate_x", "prev_1_plate_z"]

        scaler = StandardScaler()
        X_train[numeric_to_scale] = scaler.fit_transform(X_train[numeric_to_scale])
        X_test[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)[CLASSES]
        y_pred = proba.idxmax(axis=1)

        metrics = evaluate(y_test, proba, CLASSES)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        pred_freq = predicted_class_frequencies(y_pred, CLASSES)
        per_class = per_class_report(y_test, y_pred, CLASSES)
        cov = confidence_coverage(y_test, proba, CLASSES, THRESHOLDS)

        print("=" * 78)
        print(f"Step: {name}  (n_features={X_train.shape[1]})")
        print(f"  accuracy={metrics['accuracy']:.4f}  balanced_accuracy={bal_acc:.4f}  log_loss={metrics['log_loss']:.4f}")
        print()
        print("  Per-class precision / recall / F1 / support:")
        print(per_class.to_string())
        print()
        print("  Predicted class frequencies:")
        print(pred_freq.to_string())
        print()
        print("  Confidence vs. coverage:")
        print(cov.to_string(index=False))
        print()

        summary_rows.append(
            {
                "step": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": bal_acc,
                "log_loss": metrics["log_loss"],
                "SL_predicted_freq": pred_freq["SL"],
                "SL_recall": per_class.loc["SL", "recall"],
                "OTHER_predicted_freq": pred_freq["OTHER"],
            }
        )

    print("=" * 78)
    print("Summary across steps:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
