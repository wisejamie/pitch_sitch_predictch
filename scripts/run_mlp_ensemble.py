"""Probability ensemble of the 10 predetermined-seed MLPs vs. the
logistic (stand x strikes) baseline and the individual seeds.

p_ensemble(y=k|x) = (1/10) * sum_{s=0}^{9} p_s(y=k|x)

Uses the already-selected MLP architecture frozen in pitch_sitch.mlp
(SELECTED_CONFIG: hidden_layer_sizes=(32,16), alpha=0.001, epochs=19) --
this script does NOT call select_hyperparameters and does not tune
anything against the 33-game development set. Same dataset, split,
target classes, feature set, and unweighted objective as every prior
logistic/MLP experiment in this project.

First experiment to write a structured run artifact under
artifacts/experiments/<run-id>/ -- see artifacts/README.md.

Usage:
    PYTHONPATH=. python3 scripts/run_mlp_ensemble.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_sitch.artifacts import (
    git_info,
    launch_command,
    make_run_dir,
    package_versions,
    short_hash,
    utc_timestamp,
    write_json,
)
from pitch_sitch.baseline import evaluate
from pitch_sitch.design_matrix import fit_location_means
from pitch_sitch.evaluation import (
    binary_discrimination,
    bootstrap_compare_game_level,
    brier_score,
    class_calibration,
    confidence_coverage,
    predicted_class_frequencies,
)
from pitch_sitch.mlp import (
    CLASSES,
    GRID,
    INNER_SPLIT_SEED,
    INNER_SPLIT_TEST_FRAC,
    REFERENCE_SEED,
    SELECTED_CONFIG,
    SELECTION_SEED,
    SEEDS,
    build_features,
    fit_seed_ensemble,
)
from pitch_sitch.models import fit_logistic, predict_proba_df, scale_numeric
from pitch_sitch.sequence_pipeline import RICHEST_FLAGS, load_sequence_data, numeric_columns_for

THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]
DEFAULT_RUN_ID = "20260713-mlp-seed-ensemble-v1"


def average_probabilities(proba_dict: dict) -> pd.DataFrame:
    """Arithmetic mean of probability DataFrames (not a label vote)."""
    stacked = np.stack([p[CLASSES].to_numpy() for p in proba_dict.values()], axis=0)
    mean_arr = stacked.mean(axis=0)
    return pd.DataFrame(mean_arr, columns=CLASSES)


def compute_full_metrics(y_true, proba: pd.DataFrame, thresholds=THRESHOLDS, n_cal_bins: int = 5) -> dict:
    pred = proba[CLASSES].idxmax(axis=1)
    base = evaluate(y_true, proba, CLASSES)
    bs = brier_score(y_true, proba, CLASSES)
    pred_freq = predicted_class_frequencies(pred, CLASSES)

    per_class = {}
    for c in CLASSES:
        disc = binary_discrimination(y_true, proba[c], c)
        cal = class_calibration(y_true, proba[c], c, n_bins=n_cal_bins).reset_index(drop=True)
        per_class[c] = {
            "roc_auc": disc["roc_auc"],
            "pr_auc": disc["pr_auc"],
            "n_positive": disc["n_positive"],
            "calibration_mean_abs_gap": float(cal["gap"].abs().mean()),
            "calibration_bins": [
                {
                    "n": int(row["n"]),
                    "mean_predicted": float(row["mean_predicted"]),
                    "observed_rate": float(row["observed_rate"]),
                    "gap": float(row["gap"]),
                }
                for _, row in cal.iterrows()
            ],
        }

    cov = confidence_coverage(y_true, proba, CLASSES, thresholds)
    coverage_table = [
        {
            "threshold": float(r["threshold"]),
            "n_covered": int(r["n_covered"]),
            "coverage": float(r["coverage"]),
            "accuracy_at_threshold": float(r["accuracy_at_threshold"]) if pd.notna(r["accuracy_at_threshold"]) else None,
            "mean_confidence": float(r["mean_confidence"]) if pd.notna(r["mean_confidence"]) else None,
            "calibration_gap": float(r["calibration_gap"]) if pd.notna(r["calibration_gap"]) else None,
        }
        for _, r in cov.iterrows()
    ]

    return {
        "accuracy": base["accuracy"],
        "log_loss": base["log_loss"],
        "brier_score": bs,
        "n": base["n"],
        "predicted_class_frequencies": pred_freq.to_dict(),
        "per_class": per_class,
        "confidence_coverage": coverage_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP seed-probability ensemble vs. logistic baseline.")
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/raw/kevin_gausman_2022-01-01_2026-07-12.parquet")
    )
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/game_split.csv"))
    parser.add_argument("--score-diff-bound", type=int, default=6)
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--experiments-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Captured before anything else runs: writing artifacts later in this
    # script will itself dirty the tree, so git state must be recorded at
    # invocation time, not at write time, or it would misreport a clean
    # invocation as dirty.
    git_state_at_start = git_info()

    # ---- Data / features (identical pipeline to every prior logistic/MLP experiment) ----
    train, test = load_sequence_data(args.cache_file, args.split_file, args.score_diff_bound)
    location_means = fit_location_means(train, ["prev_1_plate_x", "prev_1_plate_z"])
    numeric_cols = numeric_columns_for(RICHEST_FLAGS)
    y_train, y_test = train["pitch_class"], test["pitch_class"]

    dup_ids = test.duplicated(subset=["game_pk", "at_bat_number", "pitch_number"]).sum()
    assert dup_ids == 0, f"(game_pk, at_bat_number, pitch_number) is not a unique pitch identifier: {dup_ids} duplicates"
    print(f"Pitch identifier uniqueness verified: 0 duplicates among {len(test)} development rows.\n")

    X_train = build_features(train, location_means)
    X_test = build_features(test, location_means)
    X_train_s, X_test_s = scale_numeric(X_train, X_test, numeric_cols)
    feature_names = list(X_train_s.columns)
    feature_names_hash = short_hash(",".join(feature_names))
    print(f"Feature set: {len(feature_names)} features, hash={feature_names_hash}\n")

    # ---- Logistic baseline ----
    lr_model = fit_logistic(X_train_s, y_train, class_weight=None)
    proba_lr = predict_proba_df(lr_model, X_test_s, CLASSES)

    # ---- 10 predetermined-seed MLPs, frozen architecture, no re-selection ----
    print(f"Fitting {len(SEEDS)} MLPs with frozen config {SELECTED_CONFIG} (no hyperparameter search here)...")
    proba_by_seed = fit_seed_ensemble(
        X_train_s, y_train, X_test_s,
        SELECTED_CONFIG["hidden_layer_sizes"], SELECTED_CONFIG["alpha"], SELECTED_CONFIG["best_epoch"],
        seeds=SEEDS,
    )
    for s in SEEDS:
        m = evaluate(y_test, proba_by_seed[s], CLASSES)
        print(f"  seed={s}  accuracy={m['accuracy']:.4f}  log_loss={m['log_loss']:.4f}")

    proba_reference = proba_by_seed[REFERENCE_SEED]
    proba_ensemble = average_probabilities(proba_by_seed)

    # ---- Validation checks ----
    print("\nValidation checks:")
    for name, p in [("logistic", proba_lr), ("ensemble", proba_ensemble)] + [
        (f"seed_{s}", proba_by_seed[s]) for s in SEEDS
    ]:
        row_sums = p[CLASSES].sum(axis=1).to_numpy()
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"{name}: probabilities do not sum to 1 (max dev {np.abs(row_sums-1).max()})"
    print("  All probability rows (logistic, each seed, ensemble) sum to ~1.")

    n_rows = {name: len(p) for name, p in [("test", test), ("logistic", proba_lr), ("ensemble", proba_ensemble)]}
    assert len(set(n_rows.values())) == 1, f"Row-count mismatch across models: {n_rows}"
    print(f"  All models evaluated on the identical {len(test)} development rows, same order (shared X_test_s / test index).")

    manual_ensemble = sum(proba_by_seed[s][CLASSES].to_numpy() for s in SEEDS) / len(SEEDS)
    assert np.allclose(manual_ensemble, proba_ensemble[CLASSES].to_numpy()), "Ensemble is not the arithmetic mean of seed probabilities"
    print("  Ensemble verified as the arithmetic mean of the 10 seeds' probabilities (not a label vote).")

    print(
        "  No development label was used to choose architecture, epochs, seeds, or weights: "
        "this script imports only build_features/fit_seed_ensemble from pitch_sitch.mlp, "
        "never select_hyperparameters; SELECTED_CONFIG and SEEDS are fixed constants."
    )

    # ---- Metrics ----
    metrics = {
        "logistic_baseline": compute_full_metrics(y_test, proba_lr),
        "mlp_seeds": {str(s): compute_full_metrics(y_test, proba_by_seed[s]) for s in SEEDS},
        "ensemble": compute_full_metrics(y_test, proba_ensemble),
    }
    metrics["mlp_reference_seed"] = {"seed": REFERENCE_SEED, **metrics["mlp_seeds"][str(REFERENCE_SEED)]}

    seed_accs = np.array([metrics["mlp_seeds"][str(s)]["accuracy"] for s in SEEDS])
    seed_lls = np.array([metrics["mlp_seeds"][str(s)]["log_loss"] for s in SEEDS])
    metrics["mlp_seed_distribution"] = {
        "accuracy": {"mean": float(seed_accs.mean()), "std": float(seed_accs.std()), "min": float(seed_accs.min()), "max": float(seed_accs.max())},
        "log_loss": {"mean": float(seed_lls.mean()), "std": float(seed_lls.std()), "min": float(seed_lls.min()), "max": float(seed_lls.max())},
    }

    print("\nGame-level bootstrap (n=1000), ensemble vs. logistic and vs. reference seed:")
    boot_vs_logistic = bootstrap_compare_game_level(
        test, {"logistic": proba_lr, "ensemble": proba_ensemble}, y_test, CLASSES, baseline_name="logistic", n_boot=1000, seed=0
    )
    boot_vs_reference = bootstrap_compare_game_level(
        test, {"reference_seed": proba_reference, "ensemble": proba_ensemble}, y_test, CLASSES, baseline_name="reference_seed", n_boot=1000, seed=0
    )
    metrics["game_level_bootstrap"] = {"ensemble_vs_logistic": boot_vs_logistic["ensemble"], "ensemble_vs_reference_seed": boot_vs_reference["ensemble"]}
    print(f"  ensemble vs logistic:      {boot_vs_logistic['ensemble']}")
    print(f"  ensemble vs reference_seed: {boot_vs_reference['ensemble']}")

    # SL/OTHER check -- the first time this is verified for the MLP family
    print("\nSL/OTHER top-1 prediction check:")
    for name, freqs in [("logistic", metrics["logistic_baseline"]["predicted_class_frequencies"])] + [
        (f"seed_{s}", metrics["mlp_seeds"][str(s)]["predicted_class_frequencies"]) for s in SEEDS
    ] + [("ensemble", metrics["ensemble"]["predicted_class_frequencies"])]:
        print(f"  {name:<10} SL={freqs['SL']:.4f}  OTHER={freqs['OTHER']:.4f}")

    # ---- Artifacts ----
    run_dir = make_run_dir(args.experiments_root, args.run_id, overwrite=args.overwrite)
    print(f"\nWriting artifacts to {run_dir}/")

    config = {
        "experiment_name": "mlp_seed_ensemble",
        "model_architecture": {
            "hidden_layer_sizes": list(SELECTED_CONFIG["hidden_layer_sizes"]),
            "activation": "relu",
            "solver": "adam",
        },
        "regularization": {"alpha": SELECTED_CONFIG["alpha"]},
        "epoch_count": SELECTED_CONFIG["best_epoch"],
        "seeds": SEEDS,
        "reference_seed": REFERENCE_SEED,
        "ensemble_method": "arithmetic_mean_of_probabilities",
        "target_classes": CLASSES,
        "feature_set": {
            "description": "richest sequencing + stand + stand_x_strikes (pitch_sitch.mlp.build_features)",
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "feature_names_hash": feature_names_hash,
        },
        "split": {
            "split_file": str(args.split_file),
            "split_file_hash": short_hash(Path(args.split_file).read_text()),
            "kind": "random game-level (131 train / 33 development games)",
        },
        "hyperparameter_selection_provenance": {
            "note": "This experiment freezes the configuration below rather than re-running selection. "
                    "Selection was originally performed by scripts/run_mlp_experiment.py.",
            "grid": [{"hidden_layer_sizes": list(g["hidden_layer_sizes"]), "alpha": g["alpha"]} for g in GRID],
            "inner_split_seed": INNER_SPLIT_SEED,
            "inner_split_test_frac": INNER_SPLIT_TEST_FRAC,
            "selection_seed": SELECTION_SEED,
            "selected_config": {
                "hidden_layer_sizes": list(SELECTED_CONFIG["hidden_layer_sizes"]),
                "alpha": SELECTED_CONFIG["alpha"],
                "best_epoch": SELECTED_CONFIG["best_epoch"],
            },
        },
    }

    data_summary = {
        "train_rows": len(train),
        "dev_rows": len(test),
        "train_games": int(train["game_pk"].nunique()),
        "dev_games": int(test["game_pk"].nunique()),
        "class_counts_train": y_train.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "class_counts_dev": y_test.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "date_range": [str(pd.concat([train["game_date"], test["game_date"]]).min()),
                        str(pd.concat([train["game_date"], test["game_date"]]).max())],
        "n_features": len(feature_names),
    }

    run_info = {
        "run_id": args.run_id,
        "timestamp_utc": utc_timestamp(),
        "script": "scripts/run_mlp_ensemble.py",
        "git_commit_at_start": git_state_at_start["commit"],
        "git_dirty_at_start": git_state_at_start["dirty"],
        "launch": launch_command(),
        "python_version": f"{__import__('sys').version}",
        "package_versions": package_versions(["sklearn", "pandas", "numpy"]),
        "status": "complete",
    }

    id_cols = test[["game_pk", "at_bat_number", "pitch_number"]].reset_index(drop=True)
    pitch_id = (
        id_cols["game_pk"].astype(str) + "_" + id_cols["at_bat_number"].astype(str) + "_" + id_cols["pitch_number"].astype(str)
    )
    predictions = pd.DataFrame({"pitch_id": pitch_id})
    predictions = pd.concat([predictions, id_cols], axis=1)
    predictions["actual_class"] = y_test.reset_index(drop=True)
    for c in CLASSES:
        predictions[f"logistic_p_{c}"] = proba_lr[c].reset_index(drop=True)
    for s in SEEDS:
        for c in CLASSES:
            predictions[f"seed{s}_p_{c}"] = proba_by_seed[s][c].reset_index(drop=True)
    for c in CLASSES:
        predictions[f"ensemble_p_{c}"] = proba_ensemble[c].reset_index(drop=True)
    predictions["logistic_top_pred"] = proba_lr[CLASSES].idxmax(axis=1).reset_index(drop=True)
    predictions["ensemble_top_pred"] = proba_ensemble[CLASSES].idxmax(axis=1).reset_index(drop=True)

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "data_summary.json", data_summary)
    write_json(run_dir / "run_info.json", run_info)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    # ---- Recompute-from-artifact validation ----
    reloaded = pd.read_parquet(run_dir / "predictions.parquet")
    reloaded_ensemble_proba = reloaded[[f"ensemble_p_{c}" for c in CLASSES]].rename(columns=lambda x: x.replace("ensemble_p_", ""))
    recomputed = evaluate(reloaded["actual_class"], reloaded_ensemble_proba, CLASSES)
    assert abs(recomputed["accuracy"] - metrics["ensemble"]["accuracy"]) < 1e-9
    assert abs(recomputed["log_loss"] - metrics["ensemble"]["log_loss"]) < 1e-9
    print(f"\nRecompute-from-artifact check passed: ensemble accuracy/log_loss recomputed from "
          f"predictions.parquet match metrics.json exactly.")

    print(f"\nDone. Artifacts written to {run_dir}/")


if __name__ == "__main__":
    main()
