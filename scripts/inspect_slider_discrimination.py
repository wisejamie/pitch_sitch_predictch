"""Does the model discriminate sliders even though it never predicts them?

For the count_game model and the richest sequencing model (both
unweighted, same as previously reported), inspects the P(SL) score in
isolation from the argmax decision rule: P(SL) for true SL vs. non-SL
pitches, ROC-AUC/PR-AUC for SL as a one-vs-rest problem, top-2 recall
per class, SL calibration, and per-class log loss. No model or feature
changes here -- this only re-evaluates the two models already reported.
"""

import argparse
from pathlib import Path

from pitch_sitch.evaluation import (
    binary_discrimination,
    class_calibration,
    class_probability_quantiles,
    per_class_log_loss,
    topk_recall_per_class,
)
from pitch_sitch.models import fit_count_game_logistic, load_clean_split_data, predict_proba_df
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, fit_sequence_logistic, load_sequence_data

CLASSES = ["FF", "FS", "SL", "OTHER"]


def report(name: str, y_test, proba) -> None:
    print("=" * 78)
    print(f"Model: {name}")
    print()

    disc = binary_discrimination(y_test, proba["SL"], "SL")
    print(f"  SL ROC-AUC: {disc['roc_auc']:.4f}   SL PR-AUC: {disc['pr_auc']:.4f}   (n_true_SL={disc['n_positive']})")
    print(f"  (PR-AUC baseline for a random/no-skill score = SL's base rate = {disc['n_positive'] / len(y_test):.4f})")
    print()

    print("  P(SL) quantiles, true SL vs. non-SL pitches:")
    print(class_probability_quantiles(y_test, proba["SL"], "SL").to_string())
    print()

    topk = topk_recall_per_class(y_test, proba, CLASSES, k=2)
    print("  Top-2 recall per true class (is the true class in the model's top-2 predictions?):")
    print(topk.to_string())
    print()

    print("  SL calibration (predicted P(SL) bin vs. observed SL rate):")
    print(class_calibration(y_test, proba["SL"], "SL", n_bins=5).to_string())
    print()

    print("  Per-class mean log loss (-log P(true class)):")
    print(per_class_log_loss(y_test, proba, CLASSES).to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect slider discrimination independent of argmax.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    args = parser.parse_args()

    # count_game model
    train_cg, test_cg = load_clean_split_data(args.cache_file, args.split_file, args.score_diff_bound)
    model_cg, X_train_cg, X_test_cg = fit_count_game_logistic(train_cg, test_cg, class_weight=None)
    proba_cg = predict_proba_df(model_cg, X_test_cg, CLASSES)
    report("count_game (unweighted)", test_cg["pitch_class"], proba_cg)

    # richest sequencing model (+prev3_class)
    train_seq, test_seq = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    model_seq, X_train_seq, X_test_seq = fit_sequence_logistic(train_seq, test_seq, RICHEST_FLAGS, class_weight=None)
    proba_seq = predict_proba_df(model_seq, X_test_seq, CLASSES)
    report("richest sequencing: +prev3_class (unweighted)", test_seq["pitch_class"], proba_seq)


if __name__ == "__main__":
    main()
