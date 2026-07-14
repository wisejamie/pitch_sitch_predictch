"""Export (situation, prediction, actual result) trios for the ensemble's
highest-confidence pitches on the development set, for manual inspection.

Joins the frozen ensemble predictions
(artifacts/experiments/20260713-mlp-seed-ensemble-v1/predictions.parquet)
back onto the human-readable pre-pitch situation (count, inning, outs,
baserunners, batter handedness, previous pitch) via the exact same
load_sequence_data pipeline used everywhere else, so the situation
columns can't drift from what the model actually saw. Read-only --
does not refit or change anything.
"""

import argparse
from pathlib import Path

import pandas as pd

CLASSES = ["FF", "FS", "SL", "OTHER"]

SITUATION_COLS = [
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes", "stand",
    "on_1b", "on_2b", "on_3b", "score_diff",
    "prev_1_pitch_class", "prev_1_result",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export situation/prediction/actual trios for the ensemble's highest-confidence pitches."
    )
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/experiments/20260713-mlp-seed-ensemble-v1/predictions.parquet"),
    )
    parser.add_argument("--threshold", type=float, default=0.75,
                         help="Minimum ensemble top-class probability to include (default 0.75, matching "
                              "the README's 'Very high confidence' row).")
    parser.add_argument("--out", type=Path, default=Path("data/processed/high_confidence_predictions.csv"))
    args = parser.parse_args()

    from pitch_sitch.sequence_pipeline import load_sequence_data

    _, test = load_sequence_data(args.cache_file, args.split_file)
    preds = pd.read_parquet(args.predictions)

    df = test[SITUATION_COLS].merge(
        preds, on=["game_pk", "at_bat_number", "pitch_number"], how="inner", validate="one_to_one"
    )

    proba_cols = [f"ensemble_p_{c}" for c in CLASSES]
    df["ensemble_confidence"] = df[proba_cols].max(axis=1)
    df["correct"] = df["ensemble_top_pred"] == df["actual_class"]

    high_conf = df[df["ensemble_confidence"] >= args.threshold].sort_values(
        "ensemble_confidence", ascending=False
    ).reset_index(drop=True)

    out_cols = SITUATION_COLS + ["ensemble_top_pred", "ensemble_confidence"] + proba_cols + ["actual_class", "correct"]
    out_df = high_conf[out_cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    n = len(out_df)
    acc = float(out_df["correct"].mean()) if n else float("nan")
    print(f"Threshold: ensemble confidence >= {args.threshold}")
    print(f"Pitches matching: {n} ({n / len(df) * 100:.1f}% of the {len(df)}-pitch development set)")
    print(f"Accuracy on this subset: {acc:.4f}")
    print(f"Saved full trio table to {args.out}\n")

    print("Situation | prediction (confidence) | actual | correct?")
    print("-" * 100)
    for _, row in out_df.iterrows():
        situation = (
            f"{row['game_date']} g{row['game_pk']} ab{row['at_bat_number']}.{row['pitch_number']} | "
            f"inn {row['inning']}{row['inning_topbot'][0]} out{row['outs_when_up']} | "
            f"{row['balls']}-{row['strikes']} vs {row['stand']} | "
            f"on: {int(row['on_1b']>0)}{int(row['on_2b']>0)}{int(row['on_3b']>0)} scorediff {row['score_diff']:+d} | "
            f"prev {row['prev_1_pitch_class']}/{row['prev_1_result']}"
        )
        mark = "correct" if row["correct"] else "WRONG"
        print(f"{situation}")
        print(f"    -> predicted {row['ensemble_top_pred']} ({row['ensemble_confidence']*100:.1f}%)"
              f"  |  actual {row['actual_class']}  |  {mark}")


if __name__ == "__main__":
    main()
