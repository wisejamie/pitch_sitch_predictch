"""First data audit: what does pitch-level Statcast data for one pitcher look like?

Pulls a pitcher's career Statcast pitch log via pybaseball, enforces
chronological pitch ordering, and reports pitch counts, pitch-type
frequencies by season, missingness, and basic key-uniqueness checks.

This script only describes the data. It does not select a target,
build features, or draw modelling conclusions.
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher

from pitch_sitch.features import add_runner_flags

ORDER_COLS = ["game_pk", "at_bat_number", "pitch_number"]

PRE_PITCH_COLS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b", "home_score", "away_score",
    "stand", "p_throws", "batter", "pitcher",
]

# Candidate prediction targets: what pitch was thrown and where it went.
# This project predicts the pitch, not what happened to it afterward.
TARGET_CANDIDATE_COLS = ["pitch_type", "pitch_name", "plate_x", "plate_z", "zone"]

# Not targets, but the raw material for previous-pitch context features
# (e.g. "was the last pitch taken for a ball/strike, swung at, fouled off").
# Batted-ball outcome fields (events, launch_speed, hit_location, bb_type,
# woba_value, ...) are out of scope and not audited here.
SEQUENCE_CONTEXT_COLS = ["type", "description"]


def fetch_pitcher_statcast(name: str, start_dt: str, end_dt: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_")
    cache_path = cache_dir / f"{slug}_{start_dt}_{end_dt}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    first, _, last = name.partition(" ")
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        raise ValueError(f"No player found for name={name!r}")
    if len(lookup) > 1:
        raise ValueError(
            f"Ambiguous player name={name!r}; found {len(lookup)} matches. "
            "Pass a more specific name or extend this script to accept an id."
        )
    player_id = int(lookup.iloc[0]["key_mlbam"])

    df = statcast_pitcher(start_dt, end_dt, player_id)
    df.to_parquet(cache_path)
    return df


def order_pitches(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(ORDER_COLS, ascending=True).reset_index(drop=True)


def report_pa_and_game_counts(df: pd.DataFrame) -> dict:
    return {
        "pitches": len(df),
        "plate_appearances": df.groupby(["game_pk", "at_bat_number"]).ngroups,
        "games": df["game_pk"].nunique(),
    }


def check_duplicate_pitch_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=ORDER_COLS).sum())


def report_pitch_counts_by_season(df: pd.DataFrame) -> pd.Series:
    return df.groupby("game_year").size().sort_index()


def report_pitch_type_frequency(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["game_year", "pitch_type"]).size().unstack(fill_value=0)
    counts["_total"] = counts.sum(axis=1)
    return counts


def report_missingness(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [c for c in columns if c in df.columns]
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        print(f"  (columns not present in this pull, skipped: {missing_cols})")
    return df[present].isna().mean().sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitch-level Statcast data audit for a single pitcher.")
    parser.add_argument("--name", default="Kevin Gausman", help="Pitcher full name, 'First Last'.")
    parser.add_argument("--start", default="2013-01-01", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=str(date.today()), help="End date, YYYY-MM-DD.")
    parser.add_argument("--cache-dir", default="data/raw", type=Path, help="Local cache directory for raw pulls.")
    args = parser.parse_args()

    df = fetch_pitcher_statcast(args.name, args.start, args.end, args.cache_dir)
    df = order_pitches(df)
    df = add_runner_flags(df)

    print(f"=== Pitch-level data audit: {args.name} ({args.start} to {args.end}) ===\n")

    counts = report_pa_and_game_counts(df)
    print(f"Pitches: {counts['pitches']}")
    print(f"Plate appearances: {counts['plate_appearances']}")
    print(f"Games: {counts['games']}\n")

    dup_keys = check_duplicate_pitch_keys(df)
    print(f"Duplicate ({', '.join(ORDER_COLS)}) rows: {dup_keys}\n")

    print("Pitches by season:")
    print(report_pitch_counts_by_season(df).to_string())
    print()

    print("Pitch-type frequency by season:")
    print(report_pitch_type_frequency(df).to_string())
    print()

    print("Missingness among pre-pitch context columns:")
    print(report_missingness(df, PRE_PITCH_COLS).to_string())
    print()

    print("Missingness among candidate target columns (pitch type / location):")
    print(report_missingness(df, TARGET_CANDIDATE_COLS).to_string())
    print()

    print("Missingness among sequence-context columns (feed future previous-pitch features):")
    print(report_missingness(df, SEQUENCE_CONTEXT_COLS).to_string())


if __name__ == "__main__":
    main()
