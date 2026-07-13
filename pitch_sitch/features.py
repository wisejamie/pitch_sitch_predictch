"""Pre-pitch context feature construction.

Small, composable transforms applied to the raw Statcast pull before it
is grouped by feature-group for a baseline model. Each function returns
a modified copy and is safe to call unconditionally even if its output
columns end up unused by a given feature group.
"""

import pandas as pd

RUNNER_COLS = ["on_1b", "on_2b", "on_3b"]


def add_runner_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recast on_1b/2b/3b from (player id or NaN) to a 0/1 occupied flag."""
    df = df.copy()
    for col in RUNNER_COLS:
        df[col] = df[col].notna().astype(int)
    return df


def add_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Score differential from the pitching team's perspective (fld_score - bat_score)."""
    df = df.copy()
    df["score_diff"] = df["fld_score"] - df["bat_score"]
    return df
