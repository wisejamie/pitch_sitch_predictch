"""Current-game workload features: running pitch counts, times through
the order, and prior same-game plate appearances against the batter.

All computed from information strictly before the target pitch (running
counts) or strictly before the target plate appearance (times-through-
order, prior-matchup count). Built on the raw pitch log, consistent with
pitch_sitch.sequence_features, so a pitch-clock violation (no pitch
actually thrown) isn't counted as a pitch.

times_through_order counts ALL plate appearances faced so far this game
(any batter) in groups of 9, approximating which pass through the
lineup this is. prior_pa_vs_batter counts prior plate appearances
specifically against the SAME batter this game. These are deliberately
different constructs: the first is a lineup-cycle/fatigue signal, the
second is a batter-specific familiarity signal.
"""

import pandas as pd

ORDER_COLS = ["game_pk", "at_bat_number", "pitch_number"]
NOT_A_PITCH_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}


def add_workload_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(ORDER_COLS).reset_index(drop=True)
    df["_is_real_pitch"] = (~df["description"].isin(NOT_A_PITCH_DESCRIPTIONS)).astype(int)

    df["game_pitch_count"] = df.groupby("game_pk")["_is_real_pitch"].cumsum() - df["_is_real_pitch"]
    df["inning_pitch_count"] = (
        df.groupby(["game_pk", "inning", "inning_topbot"])["_is_real_pitch"].cumsum() - df["_is_real_pitch"]
    )

    pa_level = (
        df.drop_duplicates(subset=["game_pk", "at_bat_number"])[["game_pk", "at_bat_number", "batter"]]
        .sort_values(["game_pk", "at_bat_number"])
        .reset_index(drop=True)
    )
    pa_level["pa_index_in_game"] = pa_level.groupby("game_pk").cumcount()
    pa_level["prior_pa_vs_batter"] = pa_level.groupby(["game_pk", "batter"]).cumcount()
    pa_level["times_through_order"] = 1 + (pa_level["pa_index_in_game"] // 9)

    df = df.merge(
        pa_level[["game_pk", "at_bat_number", "prior_pa_vs_batter", "times_through_order"]],
        on=["game_pk", "at_bat_number"],
        how="left",
    )
    df = df.drop(columns=["_is_real_pitch"])
    return df
