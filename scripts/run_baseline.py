"""Feature-group baseline progression for Gausman's pitch_class.

Runs the conditional-frequency lookup model for each feature group up to
and including --group, alongside the no-feature global marginal, so each
step can be compared against the previous one and against sparsity
diagnostics (how many distinct combinations appear in train, and what
fraction of test rows have to fall back to the marginal because their
exact combination was never seen in training).

This script only reports results; it does not decide whether a feature
group is "enough" or resolve sparsity issues on its own.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import (
    combo_coverage,
    evaluate,
    fit_conditional_frequency,
    fit_marginal,
    predict_proba_conditional,
    predict_proba_marginal,
)
from pitch_sitch.features import add_runner_flags, add_score_diff
from pitch_sitch.labels import build_pitch_class

CLASSES = ["FF", "FS", "SL", "OTHER"]

FEATURE_GROUPS = {
    "count": ["balls", "strikes"],
    "count_game": [
        "balls", "strikes",
        "inning", "inning_topbot", "outs_when_up", "score_diff",
        "on_1b", "on_2b", "on_3b",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-group baseline progression.")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet"),
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--group", choices=list(FEATURE_GROUPS), default="count_game")
    parser.add_argument(
        "--save-table",
        type=Path,
        default=None,
        help="Optionally save the full conditional-frequency table for --group to this CSV.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.cache_file)
    df = build_pitch_class(df)
    df = add_runner_flags(df)
    df = add_score_diff(df)

    split = pd.read_csv(args.split_file)
    df = df.merge(split, on="game_pk", how="inner")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print(f"Train pitches: {len(train)}  Test pitches: {len(test)}\n")

    marginal = fit_marginal(train, "pitch_class", CLASSES)
    baseline_proba = predict_proba_marginal(test, marginal, CLASSES)
    rows = [("global marginal", evaluate(test["pitch_class"], baseline_proba, CLASSES), None)]

    group_names = list(FEATURE_GROUPS)
    upto = group_names[: group_names.index(args.group) + 1]

    last_table = None
    for name in upto:
        feature_cols = FEATURE_GROUPS[name]
        table = fit_conditional_frequency(train, feature_cols, "pitch_class")
        proba = predict_proba_conditional(test, feature_cols, table, CLASSES, marginal)
        metrics = evaluate(test["pitch_class"], proba, CLASSES)
        coverage = combo_coverage(train, test, feature_cols)
        rows.append((name, metrics, coverage))
        if name == args.group:
            last_table = table

    header = f"{'model':<20}{'accuracy':>10}{'log_loss':>10}{'train_combos':>14}{'test_fallback%':>16}"
    print(header)
    for name, m, cov in rows:
        combos = str(cov["n_train_combos"]) if cov else "-"
        fb = f"{cov['test_fallback_rate'] * 100:.1f}" if cov else "-"
        print(f"{name:<20}{m['accuracy']:>10.4f}{m['log_loss']:>10.4f}{combos:>14}{fb:>16}")

    if args.save_table:
        args.save_table.parent.mkdir(parents=True, exist_ok=True)
        last_table.to_csv(args.save_table, index=False)
        print(f"\nSaved {args.group} conditional-frequency table ({len(last_table)} rows) to {args.save_table}")


if __name__ == "__main__":
    main()
