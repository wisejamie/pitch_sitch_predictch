"""Assign Gausman's pitch log to a random game-level train/test split.

Loads a cached raw pull (see scripts/audit_pitcher.py), applies the
accepted pitch_class label cleaning, assigns each game to train or test,
and reports resulting pitch/game/PA counts and pitch_class balance per
split so the split can be sanity-checked before it's used for modelling.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.labels import build_pitch_class
from pitch_sitch.split import assign_game_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign a random game-level train/test split.")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet"),
        help="Cached raw Statcast pull to split (see scripts/audit_pitcher.py).",
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of games held out for test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for game assignment.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/game_split.csv"),
        help="Where to save the game_pk -> split mapping.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.cache_file)
    df = build_pitch_class(df)
    df = assign_game_split(df, test_frac=args.test_frac, seed=args.seed)

    print(f"Games: {df['game_pk'].nunique()} total\n")

    print("Games per split:")
    print(df.groupby("split")["game_pk"].nunique().to_string())
    print()

    print("Pitches per split:")
    print(df["split"].value_counts().to_string())
    print()

    print("Plate appearances per split:")
    print(df.groupby("split").apply(lambda g: g.groupby(["game_pk", "at_bat_number"]).ngroups).to_string())
    print()

    print("pitch_class balance per split (should look similar between train/test):")
    print(df.groupby("split")["pitch_class"].value_counts(normalize=True).unstack().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    game_split = df[["game_pk", "split"]].drop_duplicates()
    game_split.to_csv(args.out, index=False)
    print(f"\nSaved game-level split mapping to {args.out}")


if __name__ == "__main__":
    main()
