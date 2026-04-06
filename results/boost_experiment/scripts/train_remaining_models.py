#!/usr/bin/env python3
"""
Train LightGBM, Random Forest, and MLP models for the boost experiment.
Mirrors the XGBoost training in run_boost_experiment.py with same data/splits.
"""
import gc
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

PROJECT_DIR = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

SEEDS = [42, 123, 456, 789, 1024]
CLEAN_WEIGHT = 100


def load_data():
    """Load training and test data, same as run_boost_experiment.py."""
    config = yaml.safe_load(open(PROJECT_DIR / "configs" / "training.yaml"))
    paths = config["paths"]

    # Production data
    prod_features = pd.read_parquet(PROJECT_DIR / paths["production_features"])
    prod_labels = pd.read_parquet(PROJECT_DIR / paths["production_labels"])
    prod_labels = prod_labels.set_index("_jobid")
    prod_features = prod_features.set_index("_jobid")
    common = prod_features.index.intersection(prod_labels.index)
    prod_features = prod_features.loc[common]
    prod_labels = prod_labels.loc[common]

    with open(PROJECT_DIR / paths["production_splits"], "rb") as f:
        prod_splits = pickle.load(f)
    train_idx = prod_splits.get("train_idx", prod_splits.get("train_indices"))

    # Feature columns
    exclude = set(config.get("exclude_features", []))
    for col in prod_features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_features.columns if c not in exclude]
    logger.info("Feature columns: %d", len(feature_cols))

    X_prod_train = prod_features.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_prod_train = prod_labels.iloc[train_idx][DIMENSIONS].values.astype(np.float32)
    del prod_features, prod_labels
    gc.collect()

    # Benchmark data (boost experiment) — all 689 GT samples
    splits_dir = PROJECT_DIR / "results" / "boost_experiment" / "new_splits"
    bench_features = pd.read_parquet(splits_dir / "features.parquet")
    bench_labels = pd.read_parquet(splits_dir / "labels.parquet")

    with open(PROJECT_DIR / "results" / "boost_experiment" / "new_splits" / "split_indices.pkl", "rb") as f:
        split_info = pickle.load(f)
    dev_idx = split_info.get("dev_idx", split_info.get("dev_indices", []))
    test_idx = split_info.get("test_idx", split_info.get("test_indices", []))

    # Align benchmark columns to feature_cols
    def align_cols(df, cols):
        result = []
        for col in cols:
            if col in df.columns:
                result.append(df[col].values)
            else:
                result.append(np.zeros(len(df)))
        return np.column_stack(result).astype(np.float32)

    X_bench_dev = align_cols(bench_features.iloc[dev_idx], feature_cols)
    y_bench_dev = bench_labels.iloc[dev_idx][DIMENSIONS].values.astype(np.float32)
    X_bench_test = align_cols(bench_features.iloc[test_idx], feature_cols)
    y_bench_test = bench_labels.iloc[test_idx][DIMENSIONS].values.astype(np.float32)

    # Combined training set
    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])
    weights = np.ones(len(X_combined), dtype=np.float32)
    weights[-len(X_bench_dev):] = CLEAN_WEIGHT

    logger.info("Combined training: %d samples (%d prod + %d bench dev)",
                len(X_combined), len(X_prod_train), len(X_bench_dev))
    logger.info("Test: %d samples", len(X_bench_test))

    return X_combined, y_combined, weights, X_bench_test, y_bench_test, feature_cols


def evaluate(y_true, y_pred):
    """Compute all metrics."""
    per_dim = {}
    for i, dim in enumerate(DIMENSIONS):
        per_dim[dim] = {
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "support": int(y_true[:, i].sum()),
        }
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "per_dimension": per_dim,
    }


def train_lightgbm(X_combined, y_combined, weights, X_test, y_test, output_dir):
    """Train LightGBM with 5 seeds."""
    from lightgbm import LGBMClassifier

    config = yaml.safe_load(open(PROJECT_DIR / "configs" / "training.yaml"))
    lgb_params = config["models"]["lightgbm"]["params"].copy()
    lgb_params["n_jobs"] = 4
    lgb_params["verbosity"] = -1

    spw = []
    for i in range(y_combined.shape[1]):
        n_pos = y_combined[:, i].sum()
        n_neg = len(y_combined) - n_pos
        spw.append(min(n_neg / max(n_pos, 1), 100.0))

    all_results = []
    for seed in SEEDS:
        logger.info("LightGBM seed=%d ...", seed)
        models = {}
        for i, dim in enumerate(DIMENSIONS):
            clf = LGBMClassifier(
                **lgb_params, scale_pos_weight=spw[i],
                random_state=seed,
            )
            clf.fit(X_combined, y_combined[:, i], sample_weight=weights)
            models[dim] = clf

        model_path = output_dir / f"lightgbm_biquality_w{CLEAN_WEIGHT}_seed{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

        y_pred = np.zeros_like(y_test)
        for i, dim in enumerate(DIMENSIONS):
            y_pred[:, i] = models[dim].predict(X_test)
        result = evaluate(y_test, y_pred)
        result["seed"] = seed
        all_results.append(result)
        logger.info("  Seed %d: Micro-F1=%.4f, Hamming=%.4f, SubsetAcc=%.4f",
                    seed, result["micro_f1"], result["hamming_loss"], result["subset_accuracy"])
        del models
        gc.collect()

    return all_results


def train_rf(X_combined, y_combined, weights, X_test, y_test, output_dir):
    """Train Random Forest with 5 seeds."""
    config = yaml.safe_load(open(PROJECT_DIR / "configs" / "training.yaml"))
    rf_params = config["models"].get("random_forest", {}).get("params", {}).copy()
    rf_params.setdefault("n_estimators", 500)
    rf_params.setdefault("max_depth", 15)
    rf_params["n_jobs"] = 4

    all_results = []
    for seed in SEEDS:
        logger.info("Random Forest seed=%d ...", seed)
        models = {}
        for i, dim in enumerate(DIMENSIONS):
            clf = RandomForestClassifier(
                **rf_params, random_state=seed,
            )
            clf.fit(X_combined, y_combined[:, i], sample_weight=weights)
            models[dim] = clf

        model_path = output_dir / f"random_forest_biquality_w{CLEAN_WEIGHT}_seed{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

        y_pred = np.zeros_like(y_test)
        for i, dim in enumerate(DIMENSIONS):
            y_pred[:, i] = models[dim].predict(X_test)
        result = evaluate(y_test, y_pred)
        result["seed"] = seed
        all_results.append(result)
        logger.info("  Seed %d: Micro-F1=%.4f, Hamming=%.4f, SubsetAcc=%.4f",
                    seed, result["micro_f1"], result["hamming_loss"], result["subset_accuracy"])
        del models
        gc.collect()

    return all_results


def train_mlp(X_combined, y_combined, weights, X_test, y_test, output_dir):
    """Train MLP with 5 seeds."""
    all_results = []
    for seed in SEEDS:
        logger.info("MLP seed=%d ...", seed)
        models = {}
        for i, dim in enumerate(DIMENSIONS):
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                batch_size=256,
            )
            # MLP doesn't support sample_weight in fit, so use oversampling
            # Oversample benchmark dev by weight factor
            n_prod = int((weights == 1.0).sum())
            n_bench = int((weights == CLEAN_WEIGHT).sum())
            # Repeat benchmark samples to approximate weighting
            repeat_factor = min(int(CLEAN_WEIGHT), 50)  # Cap to avoid memory issues
            X_train = np.vstack([X_combined[:n_prod], np.tile(X_combined[n_prod:], (repeat_factor, 1))])
            y_train = np.concatenate([y_combined[:n_prod, i], np.tile(y_combined[n_prod:, i], repeat_factor)])
            clf.fit(X_train, y_train)
            models[dim] = clf

        model_path = output_dir / f"mlp_biquality_w{CLEAN_WEIGHT}_seed{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

        y_pred = np.zeros_like(y_test)
        for i, dim in enumerate(DIMENSIONS):
            y_pred[:, i] = models[dim].predict(X_test)
        result = evaluate(y_test, y_pred)
        result["seed"] = seed
        all_results.append(result)
        logger.info("  Seed %d: Micro-F1=%.4f, Hamming=%.4f, SubsetAcc=%.4f",
                    seed, result["micro_f1"], result["hamming_loss"], result["subset_accuracy"])
        del models
        gc.collect()

    return all_results


def main():
    output_dir = PROJECT_DIR / "results" / "boost_experiment" / "new_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_combined, y_combined, weights, X_test, y_test, feature_cols = load_data()

    all_results = {}

    # Skip XGBoost and LightGBM (already trained from previous run)
    logger.info("Skipping XGBoost and LightGBM (already done)")

    logger.info("=" * 60)
    logger.info("Training Random Forest")
    logger.info("=" * 60)
    all_results["random_forest"] = train_rf(X_combined, y_combined, weights, X_test, y_test, output_dir)

    logger.info("=" * 60)
    logger.info("Training MLP")
    logger.info("=" * 60)
    all_results["mlp"] = train_mlp(X_combined, y_combined, weights, X_test, y_test, output_dir)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY: All Models (5 seeds, boost experiment)")
    logger.info("=" * 80)
    for model_name, results in all_results.items():
        if not results:
            continue
        mf1 = [r["micro_f1"] for r in results]
        maf1 = [r["macro_f1"] for r in results]
        hl = [r["hamming_loss"] for r in results]
        sa = [r["subset_accuracy"] for r in results]
        logger.info("%s: Micro-F1=%.3f+/-%.3f, Macro-F1=%.3f+/-%.3f, Hamming=%.3f+/-%.3f, SubsetAcc=%.3f+/-%.3f",
                    model_name,
                    np.mean(mf1), np.std(mf1),
                    np.mean(maf1), np.std(maf1),
                    np.mean(hl), np.std(hl),
                    np.mean(sa), np.std(sa))

    # Save
    results_path = output_dir.parent / "full_evaluation" / "all_models_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
