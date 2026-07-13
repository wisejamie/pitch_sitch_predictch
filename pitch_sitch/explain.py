"""Row-level explanation for a fitted linear (logistic regression) model.

Decomposes a single row's class score into per-feature contributions
(coefficient x feature value) -- the model's own arithmetic, not a
post-hoc guess about what mattered.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def class_score_contributions(model: LogisticRegression, x_row: pd.Series, target_class: str) -> pd.Series:
    class_idx = list(model.classes_).index(target_class)
    coefs = model.coef_[class_idx]
    contributions = coefs * x_row.to_numpy(dtype=float)
    return pd.Series(contributions, index=x_row.index)


def top_contributors(model: LogisticRegression, x_row: pd.Series, target_class: str, top_n: int = 3) -> pd.Series:
    contrib = class_score_contributions(model, x_row, target_class)
    nonzero = contrib[contrib != 0]
    order = nonzero.abs().sort_values(ascending=False).index[:top_n]
    return nonzero.loc[order]
