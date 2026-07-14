"""Pitch-sequencing (recent-history) feature construction.

Computes previous-pitch class, result, and location as of *before* the
current pitch, in true chronological order, resetting at plate-appearance
boundaries. History is computed on the raw (undropped) pitch log so that
a row immediately following a pitch-clock violation or an unclassified
pitch correctly gets an explicit missing history value instead of
silently referencing a non-adjacent earlier pitch. Row filtering (only
predicting resolved, classified pitches) happens after history is built.
"""

import pandas as pd

from pitch_sitch.labels import MAJOR_PITCH_TYPES, REQUIRED_COLS

ORDER_COLS = ["game_pk", "at_bat_number", "pitch_number"]
PA_GROUP_COLS = ["game_pk", "at_bat_number"]

CLASS_HISTORY_CATEGORIES = ["FF", "FS", "SL", "OTHER", "NONE"]

# Raw Statcast bat-tracking measurements. Populated only when the previous
# pitch was swung at (roughly half of swing rows even then), and only from
# 2023 onward (0% in 2022) -- see notes/decisions.md for the missingness
# audit. miss_distance is far sparser still (populated for whiffs only).
BAT_TRACKING_COLS = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "miss_distance",
]

RESULT_MAP = {
    "ball": "ball",
    "blocked_ball": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "hit_into_play": "in_play",
    "hit_by_pitch": "other",
    "missed_bunt": "other",
    # automatic_ball / automatic_strike (pitch-clock violations, no pitch
    # thrown) are intentionally absent -- they fall through to NaN and are
    # masked out below regardless, same as any other unresolved row.
}
RESULT_HISTORY_CATEGORIES = ["ball", "called_strike", "swinging_strike", "foul", "in_play", "other", "NONE"]


def assign_raw_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds pitch_class_raw and result_raw, NaN wherever the row's own
    pitch isn't a resolved, classified pitch (mirrors labels.REQUIRED_COLS)."""
    df = df.copy()
    has_required = df[REQUIRED_COLS].notna().all(axis=1)

    pitch_class = df["pitch_type"].where(df["pitch_type"].isin(MAJOR_PITCH_TYPES), "OTHER")
    df["pitch_class_raw"] = pitch_class.where(has_required)

    result = df["description"].map(RESULT_MAP)
    df["result_raw"] = result.where(has_required)
    return df


def add_history(
    df: pd.DataFrame,
    class_depths: tuple[int, ...] = (1, 2, 3),
    result_depths: tuple[int, ...] = (1,),
    location_depths: tuple[int, ...] = (1,),
    bat_tracking_depths: tuple[int, ...] = (),
) -> pd.DataFrame:
    df = df.sort_values(ORDER_COLS).reset_index(drop=True)
    grouped = df.groupby(PA_GROUP_COLS, sort=False)

    for k in class_depths:
        df[f"prev_{k}_pitch_class"] = grouped["pitch_class_raw"].shift(k).fillna("NONE")

    for k in result_depths:
        df[f"prev_{k}_result"] = grouped["result_raw"].shift(k).fillna("NONE")

    for k in location_depths:
        df[f"prev_{k}_plate_x"] = grouped["plate_x"].shift(k)
        df[f"prev_{k}_plate_z"] = grouped["plate_z"].shift(k)

    for k in bat_tracking_depths:
        for c in BAT_TRACKING_COLS:
            df[f"prev_{k}_{c}"] = grouped[c].shift(k)

    return df
