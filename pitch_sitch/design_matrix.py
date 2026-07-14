"""Design-matrix construction for the count + game-situation logistic model.

Categorical columns here have a small, fixed, rule-defined set of
possible values (count state, outs, inning half), so they are one-hot
encoded against hardcoded category lists rather than whatever happens
to appear in a given split. This guarantees train and test always
produce identically-shaped matrices.
"""

import pandas as pd

from pitch_sitch.sequence_features import BAT_TRACKING_COLS, CLASS_HISTORY_CATEGORIES, RESULT_HISTORY_CATEGORIES

BALLS_VALUES = [0, 1, 2, 3]
STRIKES_VALUES = [0, 1, 2]
COUNT_STATES = [f"{b}-{s}" for b in BALLS_VALUES for s in STRIKES_VALUES]
OUTS_VALUES = [0, 1, 2]
INNING_HALF_VALUES = ["Top", "Bot"]

NUMERIC_COLS = ["inning", "score_diff", "on_1b", "on_2b", "on_3b"]


def clip_score_diff(df: pd.DataFrame, bound: int = 6) -> pd.DataFrame:
    """Cap score_diff at +/- bound to limit the leverage of rare blowout games."""
    df = df.copy()
    df["score_diff"] = df["score_diff"].clip(-bound, bound)
    return df


def build_count_game_features(df: pd.DataFrame) -> pd.DataFrame:
    count_state = pd.Categorical(
        df["balls"].astype(str) + "-" + df["strikes"].astype(str), categories=COUNT_STATES
    )
    count_onehot = pd.get_dummies(count_state, prefix="count")

    outs = pd.Categorical(df["outs_when_up"], categories=OUTS_VALUES)
    outs_onehot = pd.get_dummies(outs, prefix="outs")

    half = pd.Categorical(df["inning_topbot"], categories=INNING_HALF_VALUES)
    half_onehot = pd.get_dummies(half, prefix="half")

    numeric = df[NUMERIC_COLS].reset_index(drop=True)

    return pd.concat(
        [
            count_onehot.reset_index(drop=True),
            outs_onehot.reset_index(drop=True),
            half_onehot.reset_index(drop=True),
            numeric,
        ],
        axis=1,
    )


def build_prev_class_onehot(df: pd.DataFrame, k: int) -> pd.DataFrame:
    cat = pd.Categorical(df[f"prev_{k}_pitch_class"], categories=CLASS_HISTORY_CATEGORIES)
    return pd.get_dummies(cat, prefix=f"prev{k}_class").reset_index(drop=True)


def build_prev_result_onehot(df: pd.DataFrame, k: int) -> pd.DataFrame:
    cat = pd.Categorical(df[f"prev_{k}_result"], categories=RESULT_HISTORY_CATEGORIES)
    return pd.get_dummies(cat, prefix=f"prev{k}_result").reset_index(drop=True)


STAND_VALUES = ["L", "R"]


def build_batter_hand_onehot(df: pd.DataFrame) -> pd.DataFrame:
    cat = pd.Categorical(df["stand"], categories=STAND_VALUES)
    return pd.get_dummies(cat, prefix="stand").reset_index(drop=True)


TTO_VALUES = [1, 2, 3, 4]
PRIOR_PA_VALUES = [0, 1, 2, 3]


def build_times_through_order_onehot(df: pd.DataFrame) -> pd.DataFrame:
    capped = df["times_through_order"].clip(upper=4)
    cat = pd.Categorical(capped, categories=TTO_VALUES)
    return pd.get_dummies(cat, prefix="tto").reset_index(drop=True)


def build_prior_pa_onehot(df: pd.DataFrame) -> pd.DataFrame:
    capped = df["prior_pa_vs_batter"].clip(upper=3)
    cat = pd.Categorical(capped, categories=PRIOR_PA_VALUES)
    return pd.get_dummies(cat, prefix="prior_pa").reset_index(drop=True)


STAND_STRIKES_STATES = [f"{s}_{k}" for s in STAND_VALUES for k in STRIKES_VALUES]
STAND_COUNT_STATES = [f"{s}_{c}" for s in STAND_VALUES for c in COUNT_STATES]


def build_stand_strikes_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Pure interaction columns (stand x strikes, 6 cells). Additive on
    top of the existing stand and count one-hots -- doesn't remove or
    replace either main effect."""
    combo = df["stand"].astype(str) + "_" + df["strikes"].astype(str)
    cat = pd.Categorical(combo, categories=STAND_STRIKES_STATES)
    return pd.get_dummies(cat, prefix="standXstrikes").reset_index(drop=True)


def build_stand_count_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Pure interaction columns (stand x full 12-state count, 24 cells).
    Additive on top of the existing stand and count one-hots."""
    combo = df["stand"].astype(str) + "_" + df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    cat = pd.Categorical(combo, categories=STAND_COUNT_STATES)
    return pd.get_dummies(cat, prefix="standXcount").reset_index(drop=True)


def fit_location_means(train_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    return {c: float(train_df[c].mean(skipna=True)) for c in cols}


def build_prev_location_numeric(df: pd.DataFrame, k: int, means: dict[str, float]) -> pd.DataFrame:
    """Numeric prev_k location, mean-imputed (train-derived) at PA start /
    missing history, plus an explicit has-value flag so the model can
    distinguish an imputed placeholder from a real observed location."""
    cols = [f"prev_{k}_plate_x", f"prev_{k}_plate_z"]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[f"{c}_has_value"] = df[c].notna().astype(int)
        out[c] = df[c].fillna(means[c])
    return out.reset_index(drop=True)


def build_prev_bat_tracking_numeric(df: pd.DataFrame, k: int, means: dict[str, float]) -> pd.DataFrame:
    """Numeric prev_k bat-tracking measurements, mean-imputed (train-
    derived) wherever missing -- PA start, pre-2023, or (much more often)
    the previous pitch simply wasn't swung at -- plus an explicit
    has-value flag per field, same pattern as build_prev_location_numeric."""
    cols = [f"prev_{k}_{c}" for c in BAT_TRACKING_COLS]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[f"{c}_has_value"] = df[c].notna().astype(int)
        out[c] = df[c].fillna(means[c])
    return out.reset_index(drop=True)
