"""How useful is the leading model (10-seed MLP ensemble) as a function
of its own confidence? Answers: "if I only trust the model's calls
above confidence X, how much of the time does it actually make a call,
and how accurate is it when it does?"

Reads the already-committed, frozen predictions from
artifacts/experiments/20260713-mlp-seed-ensemble-v1/predictions.parquet
-- no refitting, no new experiment artifact. Purely a post-hoc read of
an existing frozen result, on the same 33-game development set as
everything else in notes/current-state.md.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.evaluation import confidence_coverage

CLASSES = ["FF", "FS", "SL", "OTHER"]


def decile_table(y_true: pd.Series, max_proba: pd.Series, pred: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bin pitches by the ensemble's own top predicted probability into
    equal-sized confidence bins (not equal-width probability ranges) --
    directly answers "the top 30% most confident pitches" regardless of
    where the raw probability cutoffs fall."""
    bins = pd.qcut(max_proba, q=n_bins, duplicates="drop")
    correct = (pred.to_numpy() == y_true.to_numpy())
    table = pd.DataFrame({"bin": bins, "max_proba": max_proba, "correct": correct})
    grouped = table.groupby("bin", observed=True).agg(
        n=("correct", "size"),
        min_confidence=("max_proba", "min"),
        max_confidence=("max_proba", "max"),
        mean_confidence=("max_proba", "mean"),
        accuracy=("correct", "mean"),
    )
    grouped = grouped.sort_values("mean_confidence", ascending=False).reset_index(drop=True)
    grouped.index = [f"top {int(round((i + 1) * 100 / n_bins))}%" for i in range(len(grouped))]
    grouped["cumulative_coverage"] = [(i + 1) / n_bins for i in range(len(grouped))]
    return grouped


def by_predicted_class(y_true: pd.Series, max_proba: pd.Series, pred: pd.Series, threshold: float) -> pd.DataFrame:
    mask = max_proba.to_numpy() >= threshold
    correct = (pred.to_numpy() == y_true.to_numpy())
    d = pd.DataFrame({"pred": pred[mask].to_numpy(), "correct": correct[mask]})
    grouped = d.groupby("pred").agg(n=("correct", "size"), accuracy=("correct", "mean"))
    return grouped.reindex(CLASSES).dropna(how="all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence-vs-coverage breakdown of the leading ensemble model.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/experiments/20260713-mlp-seed-ensemble-v1/predictions.parquet"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    y_true = df["actual_class"]
    proba = df[[f"ensemble_p_{c}" for c in CLASSES]].rename(columns=lambda c: c.replace("ensemble_p_", ""))
    max_proba = proba[CLASSES].max(axis=1)
    pred = df["ensemble_top_pred"]

    overall_acc = float((pred.to_numpy() == y_true.to_numpy()).mean())
    print(f"Development set: {len(df)} pitches, {df['game_pk'].nunique()} games")
    print(f"Overall ensemble accuracy: {overall_acc:.4f}\n")

    print("=" * 100)
    print("1) Cumulative threshold view: 'trust the model whenever its top probability >= t'")
    print("=" * 100)
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    cov = confidence_coverage(y_true, proba, CLASSES, thresholds)
    cov_display = cov.copy()
    cov_display["coverage"] = (cov_display["coverage"] * 100).round(1)
    cov_display["accuracy_at_threshold"] = cov_display["accuracy_at_threshold"].round(4)
    cov_display["mean_confidence"] = cov_display["mean_confidence"].round(4)
    cov_display["calibration_gap"] = cov_display["calibration_gap"].round(4)
    print(cov_display.to_string(index=False))

    print()
    print("=" * 100)
    print("2) Equal-sized confidence-decile view: 'the top N% most confident pitches, whatever the raw threshold'")
    print("=" * 100)
    dec = decile_table(y_true, max_proba, pred, n_bins=10)
    dec_display = dec.copy()
    for col in ("min_confidence", "max_confidence", "mean_confidence", "accuracy", "cumulative_coverage"):
        dec_display[col] = dec_display[col].round(4)
    print(dec_display.to_string())

    print()
    print("=" * 100)
    print("3) Within the two most-confident deciles (top 20% by confidence), accuracy broken down by predicted class")
    print("=" * 100)
    top20_threshold = max_proba.quantile(0.80)
    print(f"(threshold for top 20%: max_proba >= {top20_threshold:.4f})\n")
    print(by_predicted_class(y_true, max_proba, pred, top20_threshold).round(4).to_string())

    print()
    print("=" * 100)
    print("4) Bottom of the confidence distribution: the least-confident third")
    print("=" * 100)
    bottom_threshold = max_proba.quantile(1 / 3)
    mask = max_proba.to_numpy() <= bottom_threshold
    acc_bottom = float((pred.to_numpy()[mask] == y_true.to_numpy()[mask]).mean())
    print(f"max_proba <= {bottom_threshold:.4f}: n={int(mask.sum())} ({mask.mean()*100:.1f}% of pitches), "
          f"accuracy={acc_bottom:.4f}")


if __name__ == "__main__":
    main()
