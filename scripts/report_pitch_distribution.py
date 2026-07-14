"""Report Gausman's observed target-class distribution on the exact
cleaned modelling dataset, and render the two summary README figures.

Uses pitch_sitch.labels.build_pitch_class -- the same target-labeling
function used by pitch_sitch/models.py and scripts/make_split.py -- so
these numbers cannot drift from what the model actually trains/evaluates
on. Does not modify any model code; this is a read-only report script.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pitch_sitch.labels import MAJOR_PITCH_TYPES, build_pitch_class

CLASSES = ["FF", "FS", "SL", "OTHER"]

# Toronto Blue Jays official brand colors (blue, red, navy) plus the
# historical throwback gray for the OTHER catch-all class.
COLORS = {"FF": "#134A8E", "FS": "#E8291C", "SL": "#1D2D5C", "OTHER": "#B1B3B3"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the cleaned target-class distribution and render figures.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.cache_file)
    n_raw = len(raw)

    df = build_pitch_class(raw)
    n_clean = len(df)
    n_dropped = n_raw - n_clean
    df["season"] = pd.to_datetime(df["game_date"]).dt.year

    print(f"Raw pitches: {n_raw}")
    print(f"Dropped (unresolved pitch_type/plate_x/plate_z/zone): {n_dropped} "
          f"({n_dropped / n_raw * 100:.2f}%)")
    print(f"Cleaned modelling dataset: {n_clean}\n")

    # 1) Overall distribution
    overall = df["pitch_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    overall_pct = (overall / n_clean * 100).round(2)
    print("=" * 80)
    print("1) Overall target-class distribution (cleaned dataset)")
    print("=" * 80)
    for c in CLASSES:
        print(f"  {c:<6} n={overall[c]:<6} {overall_pct[c]:.2f}%")

    # 2) By season
    print()
    print("=" * 80)
    print("2) By season")
    print("=" * 80)
    by_season = df.groupby(["season", "pitch_class"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    season_totals = by_season.sum(axis=1)
    by_season_pct = by_season.div(season_totals, axis=0) * 100
    print("Counts:")
    print(by_season.assign(total=season_totals).to_string())
    print("\nPercentages:")
    print(by_season_pct.round(2).to_string())

    # 3) Raw pitch types inside OTHER
    print()
    print("=" * 80)
    print("3) Raw Statcast pitch types inside OTHER")
    print("=" * 80)
    other = df[df["pitch_class"] == "OTHER"]
    print(f"Overall (n={len(other)}):")
    other_counts = other["pitch_type"].value_counts()
    other_pct = (other_counts / len(other) * 100).round(2)
    for pt in other_counts.index:
        print(f"  {pt:<6} n={other_counts[pt]:<6} {other_pct[pt]:.2f}% of OTHER  "
              f"({other_counts[pt] / n_clean * 100:.2f}% of all pitches)")

    print("\nBy season:")
    other_by_season = other.groupby(["season", "pitch_type"]).size().unstack(fill_value=0)
    print(other_by_season.to_string())

    # 4) Always-predict-FF baseline
    print()
    print("=" * 80)
    print("4) Always-predict-FF baseline accuracy")
    print("=" * 80)
    acc_full = float((df["pitch_class"] == "FF").mean())
    print(f"Full cleaned dataset (n={n_clean}): {acc_full:.4f} ({acc_full*100:.2f}%)")

    split = pd.read_csv(args.split_file)
    dev = df.merge(split, on="game_pk", how="inner")
    dev = dev[dev["split"] == "test"]
    acc_dev = float((dev["pitch_class"] == "FF").mean())
    print(f"33-game development set (n={len(dev)}): {acc_dev:.4f} ({acc_dev*100:.2f}%)")

    # Figures
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    y_pos = range(len(CLASSES))
    values = [overall_pct[c] for c in CLASSES]
    bars = ax.barh(y_pos, values, color=[COLORS[c] for c in CLASSES])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("% of pitches")
    ax.set_title(f"Kevin Gausman pitch-type distribution (n={n_clean}, 2022-01-01 to 2026-07-12)", fontsize=11)
    for bar, v in zip(bars, values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center")
    ax.set_xlim(0, max(values) + 8)
    fig.tight_layout()
    out1 = args.figures_dir / "gausman_pitch_distribution.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out1}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    seasons = by_season_pct.index.astype(str)
    bottom = pd.Series(0.0, index=by_season_pct.index)
    for c in CLASSES:
        ax.bar(seasons, by_season_pct[c], bottom=bottom, label=c, color=COLORS[c])
        bottom += by_season_pct[c]
    ax.set_ylabel("% of pitches")
    ax.set_xlabel("Season")
    ax.set_title("Kevin Gausman pitch-type distribution by season")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    out2 = args.figures_dir / "gausman_pitch_distribution_by_season.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
