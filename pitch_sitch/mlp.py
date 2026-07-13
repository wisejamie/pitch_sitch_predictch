"""Shared MLP feature-building, hyperparameter-selection, and per-seed
fitting logic for the stand x strikes feature set.

Used by scripts/run_mlp_experiment.py, which re-runs the selection
procedure (inner train/val split of the training games only, never the
development set), and by scripts/run_mlp_ensemble.py, which freezes the
already-selected configuration (SELECTED_CONFIG) and only evaluates it
-- it does not call select_hyperparameters.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import build_batter_hand_onehot, build_stand_strikes_interaction
from pitch_sitch.models import predict_proba_df
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, build_step_features

CLASSES = ["FF", "FS", "SL", "OTHER"]

GRID = [
    {"hidden_layer_sizes": (16,), "alpha": 1e-3},
    {"hidden_layer_sizes": (16,), "alpha": 1e-2},
    {"hidden_layer_sizes": (32,), "alpha": 1e-3},
    {"hidden_layer_sizes": (32,), "alpha": 1e-2},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-3},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-2},
]
MAX_EPOCHS = 300
PATIENCE = 20
INNER_SPLIT_SEED = 123
INNER_SPLIT_TEST_FRAC = 0.2
SELECTION_SEED = 0

# Frozen result of the selection procedure above, originally derived by
# scripts/run_mlp_experiment.py. Recorded here as a single source of
# truth so scripts/run_mlp_ensemble.py can evaluate an already-selected
# model without re-running the search.
SELECTED_CONFIG = {
    "hidden_layer_sizes": (32, 16),
    "alpha": 1e-3,
    "best_epoch": 19,
}

SEEDS = list(range(10))
REFERENCE_SEED = 0


def build_features(df: pd.DataFrame, location_means: dict) -> pd.DataFrame:
    return pd.concat(
        [
            build_step_features(df, RICHEST_FLAGS, location_means),
            build_batter_hand_onehot(df),
            build_stand_strikes_interaction(df),
        ],
        axis=1,
    )


def train_with_early_stopping(X_tr, y_tr, X_val, y_val, hidden_layer_sizes, alpha, seed):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )
    classes = np.array(sorted(y_tr.unique()))
    best_val_loss = np.inf
    best_epoch = 0
    since_improved = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(X_tr, y_tr, classes=classes)
        proba_val = predict_proba_df(model, X_val, CLASSES)
        val_loss = evaluate(y_val, proba_val, CLASSES)["log_loss"]
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1
        if since_improved >= PATIENCE:
            break

    return best_val_loss, best_epoch


def fit_mlp_fixed_epochs(X_tr, y_tr, hidden_layer_sizes, alpha, n_epochs, seed):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )
    classes = np.array(sorted(y_tr.unique()))
    for _ in range(n_epochs):
        model.partial_fit(X_tr, y_tr, classes=classes)
    return model


def select_hyperparameters(X_inner_train, y_inner_train, X_inner_val, y_inner_val, grid=GRID, seed=SELECTION_SEED):
    """Grid search with early stopping. Returns (best_config, all_results).
    Never touches the development set -- inner_train/inner_val must both
    come from the training games only."""
    results = []
    for cfg in grid:
        val_loss, best_epoch = train_with_early_stopping(
            X_inner_train, y_inner_train, X_inner_val, y_inner_val,
            cfg["hidden_layer_sizes"], cfg["alpha"], seed=seed,
        )
        results.append({**cfg, "val_loss": val_loss, "best_epoch": best_epoch})
    best = min(results, key=lambda r: r["val_loss"])
    return best, results


def fit_seed_ensemble(X_train, y_train, X_test, hidden_layer_sizes, alpha, n_epochs, seeds, classes=CLASSES):
    """Fits one MLP per seed on the given (already-scaled) training
    matrix. Returns {seed: proba_dataframe} -- never touches y_test."""
    proba_by_seed = {}
    for seed in seeds:
        model = fit_mlp_fixed_epochs(X_train, y_train, hidden_layer_sizes, alpha, n_epochs, seed=seed)
        proba_by_seed[seed] = predict_proba_df(model, X_test, classes)
    return proba_by_seed
