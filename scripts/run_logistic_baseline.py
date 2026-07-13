"""Multinomial logistic regression: count + game-situation features.

First non-lookup-table model for Gausman's pitch_class. Feature encoding
choices (count state one-hot, outs/inning-half one-hot, score_diff
clipped, inning left linear) are recorded in notes/decisions.md
(2026-07-12). Reports the same accuracy/log_loss metrics as the lookup
table experiments for a direct comparison.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import (
    evaluate,
    fit_conditional_frequency,
    fit_marginal,
    predict_proba_conditional,
    predict_proba_marginal,
)
from pitch_sitch.models import fit_count_game_logistic, load_clean_split_data

CLASSES = ["FF", "FS", "SL", "OTHER"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic regression baseline: count + game situation.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    train, test = load_clean_split_data(args.cache_file, args.split_file, args.score_diff_bound)
    print(f"Train pitches: {len(train)}  Test pitches: {len(test)}\n")

    y_train = train["pitch_class"]
    y_test = test["pitch_class"]

    model, X_train, X_test = fit_count_game_logistic(train, test)

    proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)[CLASSES]
    logreg_metrics = evaluate(y_test, proba, CLASSES)

    # Recompute the earlier baselines here too, so the comparison is self-contained.
    marginal = fit_marginal(train, "pitch_class", CLASSES)
    baseline_proba = predict_proba_marginal(test, marginal, CLASSES)
    baseline_metrics = evaluate(y_test, baseline_proba, CLASSES)

    count_table = fit_conditional_frequency(train, ["balls", "strikes"], "pitch_class")
    count_proba = predict_proba_conditional(test, ["balls", "strikes"], count_table, CLASSES, marginal)
    count_metrics = evaluate(y_test, count_proba, CLASSES)

    print(f"{'model':<32}{'accuracy':>10}{'log_loss':>10}")
    for name, m in [
        ("global marginal", baseline_metrics),
        ("count lookup", count_metrics),
        ("count_game logistic regression", logreg_metrics),
    ]:
        print(f"{name:<32}{m['accuracy']:>10.4f}{m['log_loss']:>10.4f}")

    print("\nPer-class coefficients (softmax score contribution, one column per class):")
    coef_df = pd.DataFrame(model.coef_, columns=X_train.columns, index=model.classes_)
    print(coef_df.T.to_string())


if __name__ == "__main__":
    main()
