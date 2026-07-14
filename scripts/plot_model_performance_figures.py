"""README figures for model performance: the confidence/coverage
trade-off of the leading ensemble, and the model-progression story
(lookup table -> logistic -> MLP ensemble).

The confidence/coverage figure is read directly from the frozen
ensemble predictions (no refitting). The model-progression figure
plots already-recorded historical numbers from notes/decisions.md and
the README's own "Model progression" table -- these predate the
structured experiment-artifact convention and are not re-derived here,
only visualized.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pitch_sitch.evaluation import confidence_coverage

CLASSES = ["FF", "FS", "SL", "OTHER"]
BLUE = "#134A8E"
RED = "#E8291C"
NAVY = "#1D2D5C"

# (label, accuracy, log_loss) -- source: notes/decisions.md, README "Model progression" table.
MODEL_PROGRESSION = [
    ("Global\ntendency", 0.521, 1.0080),
    ("Count\nlookup", 0.556, 0.9559),
    ("Count+state\nlookup", 0.505, 4.1806),
    ("Logistic\nregression", 0.563, 0.9515),
    ("+ pitch\nhistory", 0.569, 0.9371),
    ("+ handedness\ninteraction", 0.569, 0.9061),
    ("Average\nindividual MLP", 0.582, 0.8930),
    ("MLP\nensemble", 0.5853, 0.8830),
]


def plot_confidence_coverage(predictions_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(predictions_path)
    y_true = df["actual_class"]
    proba = df[[f"ensemble_p_{c}" for c in CLASSES]].rename(columns=lambda c: c.replace("ensemble_p_", ""))
    overall_acc = float((df["ensemble_top_pred"].to_numpy() == y_true.to_numpy()).mean())

    thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    cov = confidence_coverage(y_true, proba, CLASSES, thresholds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cov["threshold"] * 100, cov["accuracy_at_threshold"] * 100, marker="o", color=BLUE,
            linewidth=2, label="Accuracy on pitches at/above threshold")
    ax.plot(cov["threshold"] * 100, cov["coverage"] * 100, marker="o", color=RED,
            linewidth=2, linestyle="--", label="Coverage (% of pitches selected)")
    ax.axhline(overall_acc * 100, color=NAVY, linewidth=1, linestyle=":",
               label=f"Overall accuracy, no threshold ({overall_acc*100:.1f}%)")

    for _, row in cov.iloc[::2].iterrows():
        ax.annotate(f"n={int(row['n_covered'])}", (row["threshold"] * 100, row["coverage"] * 100),
                    textcoords="offset points", xytext=(10, 8), ha="left", fontsize=7.5, color=RED)

    ax.set_xlabel("Confidence threshold (act only when top probability ≥ t)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 108)
    ax.set_title("Ensemble accuracy vs. coverage as the confidence threshold rises\n"
                  "(33-game development set, n=3,096 pitches)", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.32), fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_model_progression(out_path: Path) -> None:
    labels = [m[0] for m in MODEL_PROGRESSION]
    acc = [m[1] * 100 for m in MODEL_PROGRESSION]
    ll = [m[2] for m in MODEL_PROGRESSION]
    x = range(len(labels))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(x, acc, marker="o", color=BLUE, linewidth=2, label="Accuracy")
    ax2.plot(x, ll, marker="s", color=RED, linewidth=2, linestyle="--", label="Log loss")

    # The count+game-state lookup table's log loss (4.18) would compress every
    # other bar's visible variation -- clip the axis and annotate it instead.
    ax2.set_ylim(0.7, 1.1)
    ax2.annotate(f"{ll[2]:.2f}", (2, 1.08), color=RED, fontsize=9, ha="center", fontweight="bold")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Accuracy (%)", color=BLUE)
    ax2.set_ylabel("Log loss (lower is better)", color=RED)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("Model progression: accuracy and log loss across the project's model history", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the model-performance README figures.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/experiments/20260713-mlp-seed-ensemble-v1/predictions.parquet"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confidence_coverage(args.predictions, args.figures_dir / "confidence_accuracy_coverage.png")
    plot_model_progression(args.figures_dir / "model_progression.png")


if __name__ == "__main__":
    main()
