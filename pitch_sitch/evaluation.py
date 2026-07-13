"""Detailed evaluation diagnostics for a classifier's test-set predictions.

Given true labels and predicted class probabilities, computes standard
multi-class diagnostics (confusion matrix, per-class precision/recall/F1,
predicted class frequencies) and confidence-vs-coverage behavior at a
set of probability thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from pitch_sitch.baseline import evaluate


def per_class_report(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=classes,
    )


def confusion(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(
        cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    )


def predicted_class_frequencies(y_pred, classes: list[str]) -> pd.Series:
    return pd.Series(y_pred).value_counts(normalize=True).reindex(classes, fill_value=0.0)


def confidence_coverage(y_true, proba: pd.DataFrame, classes: list[str], thresholds: list[float]) -> pd.DataFrame:
    """For each threshold t: among test rows where the model's top predicted
    probability is >= t, what's the accuracy, and how does mean predicted
    confidence compare to that accuracy (positive calibration_gap = overconfident)."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    max_proba = proba[classes].max(axis=1).reset_index(drop=True)
    pred = proba[classes].idxmax(axis=1).reset_index(drop=True)
    correct = (pred.to_numpy() == y_true.to_numpy())

    n_total = len(y_true)
    rows = []
    for t in thresholds:
        mask = max_proba.to_numpy() >= t
        n_covered = int(mask.sum())
        coverage = n_covered / n_total
        if n_covered > 0:
            acc_at_t = float(correct[mask].mean())
            mean_conf = float(max_proba.to_numpy()[mask].mean())
            calibration_gap = mean_conf - acc_at_t
        else:
            acc_at_t = float("nan")
            mean_conf = float("nan")
            calibration_gap = float("nan")
        rows.append(
            {
                "threshold": t,
                "n_covered": n_covered,
                "coverage": coverage,
                "accuracy_at_threshold": acc_at_t,
                "mean_confidence": mean_conf,
                "calibration_gap": calibration_gap,
            }
        )
    return pd.DataFrame(rows)


def binary_discrimination(y_true, proba_class, positive_label: str) -> dict:
    """ROC-AUC and PR-AUC for one class treated as a one-vs-rest binary
    problem, using its predicted probability as the score. Independent of
    the argmax decision rule -- measures whether the score itself
    separates the class from everything else."""
    y_binary = (pd.Series(y_true).reset_index(drop=True).to_numpy() == positive_label).astype(int)
    scores = pd.Series(proba_class).reset_index(drop=True).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(y_binary, scores)),
        "pr_auc": float(average_precision_score(y_binary, scores)),
        "n_positive": int(y_binary.sum()),
    }


def topk_recall_per_class(y_true, proba: pd.DataFrame, classes: list[str], k: int = 2) -> pd.Series:
    """For each true class, the fraction of rows where that class is
    among the model's top-k predicted classes by probability."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    arr = proba[classes].to_numpy()
    topk_idx = np.argsort(-arr, axis=1)[:, :k]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y_true.map(class_to_idx).to_numpy()
    hit = np.array([y_idx[i] in topk_idx[i] for i in range(len(y_idx))])

    result = {}
    for c in classes:
        mask = y_true.to_numpy() == c
        result[c] = float(hit[mask].mean()) if mask.sum() > 0 else float("nan")
    return pd.Series(result)


def class_probability_quantiles(
    y_true, proba_class, positive_label: str, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)
) -> pd.DataFrame:
    """Quantiles of predicted P(class) split by whether the row's true
    label actually is that class, to see how much separation exists."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    proba_class = pd.Series(proba_class).reset_index(drop=True)
    is_pos = y_true.to_numpy() == positive_label
    return pd.DataFrame(
        {
            f"true_{positive_label}": proba_class[is_pos].quantile(quantiles),
            f"non_{positive_label}": proba_class[~is_pos].quantile(quantiles),
        }
    )


def class_calibration(y_true, proba_class, positive_label: str, n_bins: int = 5) -> pd.DataFrame:
    """Reliability table for one class: within bins of predicted P(class),
    compare mean predicted probability to the actual observed rate."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    proba_class = pd.Series(proba_class).reset_index(drop=True)
    y_binary = (y_true.to_numpy() == positive_label).astype(int)
    bins = pd.qcut(proba_class, q=n_bins, duplicates="drop")
    table = pd.DataFrame({"bin": bins, "pred": proba_class, "actual": y_binary})
    grouped = table.groupby("bin", observed=True).agg(
        n=("actual", "size"), mean_predicted=("pred", "mean"), observed_rate=("actual", "mean")
    )
    grouped["gap"] = grouped["mean_predicted"] - grouped["observed_rate"]
    return grouped


def class_diagnostic_by_group(
    df: pd.DataFrame,
    y_true,
    proba_class,
    pred,
    group_cols: list[str],
    positive_label: str,
) -> pd.DataFrame:
    """For one class, grouped by arbitrary columns (e.g. count, or count x
    handedness): sample size, actual rate, mean predicted probability,
    calibration gap, and recall. Used to check whether a probability gap
    for one class is concentrated in specific situations."""
    d = df[group_cols].reset_index(drop=True).copy()
    d["actual_is_class"] = (pd.Series(y_true).reset_index(drop=True).to_numpy() == positive_label).astype(int)
    d["pred_is_class"] = (pd.Series(pred).reset_index(drop=True).to_numpy() == positive_label).astype(int)
    d["p_class"] = pd.Series(proba_class).reset_index(drop=True)

    grouped = d.groupby(group_cols)
    result = grouped.agg(
        n=("actual_is_class", "size"),
        actual_rate=("actual_is_class", "mean"),
        mean_predicted=("p_class", "mean"),
    )
    result["calibration_gap"] = result["mean_predicted"] - result["actual_rate"]

    recall = d[d["actual_is_class"] == 1].groupby(group_cols)["pred_is_class"].mean()
    result = result.join(recall.rename("recall"))
    return result.reset_index()


def brier_score(y_true, proba: pd.DataFrame, classes: list[str]) -> float:
    """Multiclass Brier score: mean over rows of the summed squared error
    between the predicted probability vector and the one-hot true-label
    vector.

        BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    where p_ik is the predicted probability of class k for row i, and
    y_ik is 1 if row i's true class is k, else 0. Lower is better; a
    perfect prediction scores 0, and always predicting the uniform
    distribution over C classes scores (C-1)/C in the worst case."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    p = proba[classes].to_numpy(dtype=float)
    one_hot = np.zeros_like(p)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = y_true.map(class_index).to_numpy()
    one_hot[np.arange(len(p)), y_idx] = 1.0
    return float(np.sum((p - one_hot) ** 2, axis=1).mean())


def bootstrap_compare_game_level(
    test_df: pd.DataFrame,
    proba_by_model: dict,
    y_true,
    classes: list[str],
    baseline_name: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Grouped (game-level) bootstrap: resamples whole games with
    replacement from test_df["game_pk"], and for each resample computes
    accuracy/log_loss for every model relative to baseline_name. Models
    are fit once beforehand -- only the evaluation set is resampled
    here, so this describes stability across which development games
    are included, not out-of-sample generalization."""
    rng = np.random.default_rng(seed)
    games = test_df["game_pk"].unique()
    n_games = len(games)
    game_idx_arr = test_df["game_pk"].to_numpy()
    idx_by_game = {g: np.where(game_idx_arr == g)[0] for g in games}

    other_names = [n for n in proba_by_model if n != baseline_name]
    diffs = {n: {"accuracy": [], "log_loss": []} for n in other_names}

    for _ in range(n_boot):
        sampled_games = rng.choice(games, size=n_games, replace=True)
        idx = np.concatenate([idx_by_game[g] for g in sampled_games])
        y_sample = pd.Series(y_true).iloc[idx].reset_index(drop=True)

        base_m = evaluate(y_sample, proba_by_model[baseline_name].iloc[idx].reset_index(drop=True), classes)
        for n in other_names:
            m = evaluate(y_sample, proba_by_model[n].iloc[idx].reset_index(drop=True), classes)
            diffs[n]["accuracy"].append(m["accuracy"] - base_m["accuracy"])
            diffs[n]["log_loss"].append(m["log_loss"] - base_m["log_loss"])

    summary = {}
    for n in other_names:
        acc = np.array(diffs[n]["accuracy"])
        ll = np.array(diffs[n]["log_loss"])
        summary[n] = {
            "accuracy_diff_mean": float(acc.mean()),
            "accuracy_diff_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "p_accuracy_improves": float(np.mean(acc > 0)),
            "log_loss_diff_mean": float(ll.mean()),
            "log_loss_diff_ci95": [float(np.percentile(ll, 2.5)), float(np.percentile(ll, 97.5))],
            "p_log_loss_improves": float(np.mean(ll < 0)),
        }
    return summary


def per_class_log_loss(y_true, proba: pd.DataFrame, classes: list[str]) -> pd.Series:
    """Mean -log(p_true_class), restricted to rows of each true class.
    Shows how much probability mass the model puts near the truth even
    when that class never wins argmax."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    eps = 1e-9
    p = np.clip(proba[classes].to_numpy(dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = y_true.map(class_index).to_numpy()
    row_loss = -np.log(p[np.arange(len(p)), y_idx])

    result = {}
    for c in classes:
        mask = y_true.to_numpy() == c
        result[c] = float(row_loss[mask].mean()) if mask.sum() > 0 else float("nan")
    return pd.Series(result)
