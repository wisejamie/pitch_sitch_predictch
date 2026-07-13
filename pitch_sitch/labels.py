"""Pitch-type target label definition for Kevin Gausman.

Implements the accepted decision recorded in notes/decisions.md
(2026-07-12): pitches outside {FF, FS, SL} are collapsed into OTHER, and
rows without a resolved pitch_type or location are dropped.
"""

import pandas as pd

MAJOR_PITCH_TYPES = ["FF", "FS", "SL"]

REQUIRED_COLS = ["pitch_type", "plate_x", "plate_z", "zone"]


def build_pitch_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["pitch_class"] = df["pitch_type"].where(df["pitch_type"].isin(MAJOR_PITCH_TYPES), "OTHER")
    return df
