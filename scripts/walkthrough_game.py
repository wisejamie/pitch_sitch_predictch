"""Pitch-by-pitch walkthrough of one representative held-out test game.

Uses the model exactly as currently decided: richest sequencing feature
set (+prev3_class), unweighted logistic regression (no class weighting),
same game-level split as every other experiment. Selects the test game
whose pitch count is closest to the median test-game pitch count, as a
simple, non-cherry-picked notion of "representative". Purely
descriptive -- no model, feature, or split changes.
"""

import argparse
from pathlib import Path

import pandas as pd

from pitch_sitch.baseline import evaluate
from pitch_sitch.explain import top_contributors
from pitch_sitch.models import predict_proba_df
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, fit_sequence_logistic, load_sequence_data

CLASSES = ["FF", "FS", "SL", "OTHER"]


def select_representative_game(test: pd.DataFrame) -> int:
    counts = test.groupby("game_pk").size()
    median = counts.median()
    return int((counts - median).abs().idxmin())


def runners_str(row) -> str:
    on = [b for b, flag in zip(["1b", "2b", "3b"], [row["on_1b"], row["on_2b"], row["on_3b"]]) if flag]
    return "+".join(on) if on else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitch-by-pitch walkthrough of one test game.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    parser.add_argument("--game-pk", type=int, default=None, help="Override game selection.")
    args = parser.parse_args()

    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    model, X_train, X_test = fit_sequence_logistic(train, test, RICHEST_FLAGS, class_weight=None)
    proba = predict_proba_df(model, X_test, CLASSES)

    counts = test.groupby("game_pk").size()
    game_pk = args.game_pk or select_representative_game(test)
    print(
        f"Selected game_pk={game_pk}  pitches={counts[game_pk]}  "
        f"(test median={counts.median():.0f}, test game range {counts.min()}-{counts.max()} pitches)"
    )

    game_mask = (test["game_pk"] == game_pk).to_numpy()
    first_idx = test.index[game_mask][0]
    print(f"game_date={test.loc[first_idx, 'game_date']}\n")

    game_metrics = evaluate(test.loc[game_mask, "pitch_class"], proba.loc[game_mask], CLASSES)
    print(
        f"Whole-test-set accuracy/log_loss for reference: "
        f"{evaluate(test['pitch_class'], proba, CLASSES)['accuracy']:.3f} / "
        f"{evaluate(test['pitch_class'], proba, CLASSES)['log_loss']:.3f}"
    )
    print(f"This game's accuracy/log_loss: {game_metrics['accuracy']:.3f} / {game_metrics['log_loss']:.3f}\n")

    current_pa = None
    seq_so_far = []
    n_correct = 0
    n_total = 0

    for pos in range(len(test)):
        if not game_mask[pos]:
            continue
        row = test.iloc[pos]

        if row["at_bat_number"] != current_pa:
            current_pa = row["at_bat_number"]
            seq_so_far = []
            print("-" * 100)
            print(
                f"PA {int(current_pa)}: inning {int(row['inning'])} {row['inning_topbot']}, "
                f"outs {int(row['outs_when_up'])}, score_diff {int(row['score_diff'])}, "
                f"batter stand {row['stand']}, runners {runners_str(row)}"
            )

        p = proba.iloc[pos]
        pred = p.idxmax()
        actual = row["pitch_class"]
        correct = pred == actual
        n_total += 1
        n_correct += int(correct)

        contrib_pred = top_contributors(model, X_test.iloc[pos], pred, top_n=3)
        contrib_str = ", ".join(f"{k}={v:+.2f}" for k, v in contrib_pred.items())

        print(
            f"  pitch {int(row['pitch_number'])}  count {int(row['balls'])}-{int(row['strikes'])}  "
            f"seq_so_far={seq_so_far}"
        )
        print(
            f"    P(FF)={p['FF']:.2f} P(FS)={p['FS']:.2f} P(SL)={p['SL']:.2f} P(OTHER)={p['OTHER']:.2f}  "
            f"pred={pred}  actual={actual}  {'OK' if correct else 'MISS'}"
        )
        print(f"    top contributors to P({pred}): {contrib_str}")

        if not correct:
            contrib_actual = top_contributors(model, X_test.iloc[pos], actual, top_n=3)
            contrib_actual_str = ", ".join(f"{k}={v:+.2f}" for k, v in contrib_actual.items())
            print(f"    top contributors to P({actual}) [the actual pitch]: {contrib_actual_str}")

        seq_so_far.append(actual)

    print("-" * 100)
    print(f"\nGame record: {n_correct}/{n_total} correct = {n_correct / n_total:.3f}")


if __name__ == "__main__":
    main()
