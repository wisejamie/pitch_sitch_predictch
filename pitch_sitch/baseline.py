"""Conditional-frequency baseline model.

Fits P(target | feature_cols) as an empirical frequency table on
training data and evaluates it on held-out data. This is the
"count-conditioned lookup model" step in the model progression in
notes/research-log.md, written generally over feature_cols so later
steps (count + game state, etc.) can reuse it without changes.
"""

import numpy as np
import pandas as pd


def fit_marginal(df: pd.DataFrame, target_col: str, classes: list[str]) -> pd.Series:
    counts = df[target_col].value_counts().reindex(classes, fill_value=0)
    return counts / counts.sum()


def fit_conditional_frequency(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    counts = df.groupby(feature_cols)[target_col].value_counts().unstack(fill_value=0)
    n_train = counts.sum(axis=1)
    probs = counts.div(n_train, axis=0)
    probs["n_train"] = n_train
    return probs.reset_index()


def predict_proba_conditional(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    prob_table: pd.DataFrame,
    classes: list[str],
    fallback: pd.Series,
) -> pd.DataFrame:
    merged = test_df[feature_cols].reset_index(drop=True).merge(prob_table, on=feature_cols, how="left")
    proba = merged[classes].copy()
    missing = proba.isna().any(axis=1)
    for c in classes:
        proba.loc[missing, c] = fallback[c]
    return proba


def predict_proba_marginal(test_df: pd.DataFrame, marginal: pd.Series, classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame([marginal[classes].to_numpy()] * len(test_df), columns=classes)


def combo_coverage(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """How many distinct feature combinations exist in train, and what
    fraction of test rows land on a combination never seen in train
    (and therefore fall back to the marginal distribution)."""
    train_combos = train_df[feature_cols].drop_duplicates()
    merged = test_df[feature_cols].merge(train_combos.assign(_seen=1), on=feature_cols, how="left")
    return {
        "n_train_combos": int(len(train_combos)),
        "test_fallback_rate": float(merged["_seen"].isna().mean()),
    }


def evaluate(y_true: pd.Series, proba: pd.DataFrame, classes: list[str]) -> dict:
    y_true = y_true.reset_index(drop=True)
    pred = proba.idxmax(axis=1)
    accuracy = float((pred.to_numpy() == y_true.to_numpy()).mean())

    eps = 1e-9
    p = np.clip(proba[classes].to_numpy(dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = y_true.map(class_index).to_numpy()
    log_loss = float(-np.log(p[np.arange(len(p)), y_idx]).mean())

    return {"accuracy": accuracy, "log_loss": log_loss, "n": int(len(y_true))}
