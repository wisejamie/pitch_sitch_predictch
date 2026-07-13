"""Game-level train/test split.

Assigns each game (game_pk) to train or test at random, so every pitch
from a given game stays on one side of the split. Decided 2026-07-12
(see notes/decisions.md) to avoid history-feature leakage within a
plate appearance.
"""

import numpy as np
import pandas as pd


def assign_game_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    games = np.sort(df["game_pk"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(games)
    n_test = round(len(shuffled) * test_frac)
    test_games = set(shuffled[:n_test])

    df = df.copy()
    df["split"] = np.where(df["game_pk"].isin(test_games), "test", "train")
    return df
