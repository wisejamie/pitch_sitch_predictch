"""Model-fitting helpers shared across scripts that need the same
count_game logistic regression, so preprocessing/fitting isn't
duplicated between the training script and inspection scripts.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pitch_sitch.design_matrix import build_count_game_features, clip_score_diff
from pitch_sitch.features import add_runner_flags, add_score_diff
from pitch_sitch.labels import build_pitch_class

NUMERIC_TO_SCALE = ["inning", "score_diff"]


def load_clean_split_data(cache_file: Path, split_file: Path, score_diff_bound: int = 6):
    df = pd.read_parquet(cache_file)
    df = build_pitch_class(df)
    df = add_runner_flags(df)
    df = add_score_diff(df)
    df = clip_score_diff(df, bound=score_diff_bound)

    split = pd.read_csv(split_file)
    df = df.merge(split, on="game_pk", how="inner")

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def scale_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_cols: list[str]):
    """Fits StandardScaler on train only, applies to both -- avoids
    leaking test-set statistics into the scaling."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test


def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series, class_weight=None) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def predict_proba_df(model: LogisticRegression, X_test: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)[classes]


def fit_count_game_logistic(train: pd.DataFrame, test: pd.DataFrame, class_weight=None):
    X_train = build_count_game_features(train)
    X_test = build_count_game_features(test)
    X_train, X_test = scale_numeric(X_train, X_test, NUMERIC_TO_SCALE)

    model = fit_logistic(X_train, train["pitch_class"], class_weight=class_weight)

    return model, X_train, X_test
