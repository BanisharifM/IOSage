"""
ML Ablation Studies for IOSage Paper

Addresses reviewer concern: "Did the ML just learn Drishti's rules?"

Three ablations:
  1. Remove derived features — train on raw counters + indicators only
  2. GT-only training — train without heuristic labels
  3. Leave-one-benchmark-out — generalization across benchmark types

Usage:
    python scripts/run_ml_ablations.py
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

# 39 derived features added by stage3_engineer()
DERIVED_FEATURES = [
    "access_size_concentration", "avg_read_size", "avg_write_size",
    "byte_imbalance", "collective_ratio", "consec_read_ratio",
    "consec_write_ratio", "dominant_access_size", "file_misalign_ratio",
    "fsync_ratio", "io_active_fraction", "io_duration", "large_read_ratio",
    "large_write_ratio", "medium_read_ratio", "medium_write_ratio",
    "mem_misalign_ratio", "metadata_time_ratio", "nonblocking_ratio",
    "opens_per_mb", "opens_per_op", "rank_bytes_cv", "rank_time_cv",
    "read_bw_mb_s", "read_ratio", "read_time_fraction", "rw_ratio",
    "rw_switch_ratio", "seeks_per_op", "seq_read_ratio", "seq_write_ratio",
    "small_io_ratio", "small_read_ratio", "small_write_ratio",
    "stats_per_op", "time_imbalance", "total_bw_mb_s", "write_bw_mb_s",
    "write_time_fraction",
]


def load_config():
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        return yaml.safe_load(f)


def get_feature_cols(features, config, drop_derived=False):
    """Get feature column names, optionally excluding derived features."""
    exclude = set(config.get("exclude_features", []))
    for col in features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    if drop_derived:
        exclude.update(DERIVED_FEATURES)
    feature_cols = [c for c in features.columns if c not in exclude]
    return feature_cols


def load_all_data(config, drop_derived=False):
    """Load production + benchmark data."""
    paths = config["paths"]

    # Production
    prod_feat = pd.read_parquet(PROJECT_DIR / paths["production_features"])
    prod_labels = pd.read_parquet(PROJECT_DIR / paths["production_labels"])
    prod_labels = prod_labels.set_index("_jobid")
    prod_feat = prod_feat.set_index("_jobid")
    common = prod_feat.index.intersection(prod_labels.index)
    prod_feat = prod_feat.loc[common]
    prod_labels = prod_labels.loc[common]

    feature_cols = get_feature_cols(prod_feat, config, drop_derived=drop_derived)

    X_prod = prod_feat[feature_cols].values.astype(np.float32)
    y_prod = prod_labels[DIMENSIONS].values.astype(np.float32)

    with open(PROJECT_DIR / paths["production_splits"], "rb") as f:
        splits = pickle.load(f)
    train_idx = splits.get("train_idx", splits.get("train_indices"))
    val_idx = splits.get("val_idx", splits.get("val_indices"))

    # Benchmark
    bench_dir = PROJECT_DIR / "data" / "processed" / "benchmark"
    bench_feat = pd.read_parquet(bench_dir / "features.parquet")
    bench_labels = pd.read_parquet(bench_dir / "labels.parquet")

    X_bench_cols = []
    for col in feature_cols:
        if col in bench_feat.columns:
            X_bench_cols.append(bench_feat[col].values)
        else:
            X_bench_cols.append(np.zeros(len(bench_feat)))
    X_bench = np.column_stack(X_bench_cols).astype(np.float32)
    y_bench = bench_labels[DIMENSIONS].values.astype(np.float32)

    with open(bench_dir / "split_indices.pkl", "rb") as f:
        bench_splits = pickle.load(f)
    dev_idx = bench_splits["dev_idx"]
    test_idx = bench_splits["test_idx"]

    return {
        "X_prod": X_prod, "y_prod": y_prod,
        "train_idx": train_idx, "val_idx": val_idx,
        "X_bench": X_bench, "y_bench": y_bench,
        "dev_idx": dev_idx, "test_idx": test_idx,
        "feature_cols": feature_cols,
        "bench_labels": bench_labels,
    }


def compute_scale_pos_weight(y, max_weight=100.0):
    weights = []
    for i in range(y.shape[1]):
        n_pos = y[:, i].sum()
        n_neg = len(y) - n_pos
        w = min(n_neg / max(n_pos, 1), max_weight)
        weights.append(w)
    return weights


def train_xgboost_biquality(X_train, y_train, w_train, X_val, y_val,
                             config, seed=42):
    """Train per-label XGBoost with sample weights."""
    from xgboost import XGBClassifier

    params = config["models"]["xgboost"]["params"].copy()
    spw = compute_scale_pos_weight(y_train, max_weight=100.0)
    models = {}

    for i, dim in enumerate(DIMENSIONS):
        clf = XGBClassifier(
            **params, scale_pos_weight=spw[i],
            random_state=seed, verbosity=0,
        )
        clf.fit(
            X_train, y_train[:, i],
            sample_weight=w_train,
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        models[dim] = clf

    return models


def train_xgboost_simple(X_train, y_train, X_val, y_val, config, seed=42):
    """Train per-label XGBoost without sample weights (for GT-only)."""
    from xgboost import XGBClassifier

    params = config["models"]["xgboost"]["params"].copy()
    spw = compute_scale_pos_weight(y_train, max_weight=100.0)
    models = {}

    for i, dim in enumerate(DIMENSIONS):
        clf = XGBClassifier(
            **params, scale_pos_weight=spw[i],
            random_state=seed, verbosity=0,
        )
        # Use small portion of training data for eval_set if enough samples
        clf.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        models[dim] = clf

    return models


def evaluate(models, X_test, y_test):
    """Evaluate models, return metrics dict."""
    y_pred = np.zeros_like(y_test)
    for i, dim in enumerate(DIMENSIONS):
        y_pred[:, i] = models[dim].predict(X_test)

    micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    per_label = {}
    for i, dim in enumerate(DIMENSIONS):
        per_label[dim] = {
            "f1": float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "precision": float(precision_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "support": int(y_test[:, i].sum()),
        }

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "per_label": per_label,
    }


def run_ablation1_no_derived(config, seed=42):
    """Ablation 1: Remove derived features, keep only raw counters + indicators."""
    logger.info("=" * 70)
    logger.info("ABLATION 1: Remove Derived Features")
    logger.info("=" * 70)

    results = {}

    # --- Baseline: all features ---
    logger.info("Training baseline (all features)...")
    data = load_all_data(config, drop_derived=False)
    n_feat_all = len(data["feature_cols"])

    X_train = np.vstack([data["X_prod"][data["train_idx"]],
                         data["X_bench"][data["dev_idx"]]])
    y_train = np.vstack([data["y_prod"][data["train_idx"]],
                         data["y_bench"][data["dev_idx"]]])
    weights = np.ones(len(X_train))
    weights[-len(data["dev_idx"]):] = 100.0

    X_val = data["X_prod"][data["val_idx"]]
    y_val = data["y_prod"][data["val_idx"]]
    X_test = data["X_bench"][data["test_idx"]]
    y_test = data["y_bench"][data["test_idx"]]

    models_all = train_xgboost_biquality(X_train, y_train, weights, X_val, y_val,
                                          config, seed=seed)
    metrics_all = evaluate(models_all, X_test, y_test)
    logger.info("  All features (%d): Micro-F1=%.4f, Macro-F1=%.4f",
                n_feat_all, metrics_all["micro_f1"], metrics_all["macro_f1"])
    results["all_features"] = {
        "n_features": n_feat_all,
        **metrics_all,
    }

    # --- Ablation: no derived features ---
    logger.info("Training without derived features...")
    data_nd = load_all_data(config, drop_derived=True)
    n_feat_raw = len(data_nd["feature_cols"])

    X_train_nd = np.vstack([data_nd["X_prod"][data_nd["train_idx"]],
                            data_nd["X_bench"][data_nd["dev_idx"]]])
    y_train_nd = np.vstack([data_nd["y_prod"][data_nd["train_idx"]],
                            data_nd["y_bench"][data_nd["dev_idx"]]])
    weights_nd = np.ones(len(X_train_nd))
    weights_nd[-len(data_nd["dev_idx"]):] = 100.0

    X_val_nd = data_nd["X_prod"][data_nd["val_idx"]]
    y_val_nd = data_nd["y_prod"][data_nd["val_idx"]]
    X_test_nd = data_nd["X_bench"][data_nd["test_idx"]]
    y_test_nd = data_nd["y_bench"][data_nd["test_idx"]]

    models_nd = train_xgboost_biquality(X_train_nd, y_train_nd, weights_nd,
                                         X_val_nd, y_val_nd, config, seed=seed)
    metrics_nd = evaluate(models_nd, X_test_nd, y_test_nd)
    logger.info("  Raw only (%d): Micro-F1=%.4f, Macro-F1=%.4f",
                n_feat_raw, metrics_nd["micro_f1"], metrics_nd["macro_f1"])
    results["raw_only"] = {
        "n_features": n_feat_raw,
        **metrics_nd,
    }

    # Derived features that were dropped (excluding those already excluded)
    excluded_config = set(config.get("exclude_features", []))
    actually_dropped = [f for f in DERIVED_FEATURES if f not in excluded_config]
    results["derived_features_dropped"] = actually_dropped
    results["n_derived_dropped"] = len(actually_dropped)

    delta = metrics_all["micro_f1"] - metrics_nd["micro_f1"]
    logger.info("  Delta Micro-F1: %+.4f (derived features %s)",
                delta, "help" if delta > 0 else "hurt or neutral")

    return results


def run_ablation2_gt_only(config, seed=42):
    """Ablation 2: Train on GT only (no heuristic labels)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION 2: GT-Only Training (no heuristic labels)")
    logger.info("=" * 70)

    data = load_all_data(config, drop_derived=False)

    X_bench_dev = data["X_bench"][data["dev_idx"]]
    y_bench_dev = data["y_bench"][data["dev_idx"]]
    X_bench_test = data["X_bench"][data["test_idx"]]
    y_bench_test = data["y_bench"][data["test_idx"]]

    results = {}

    # --- Baseline: biquality (heuristic + GT) ---
    logger.info("Training biquality baseline (heuristic + GT)...")
    X_train_bq = np.vstack([data["X_prod"][data["train_idx"]],
                            data["X_bench"][data["dev_idx"]]])
    y_train_bq = np.vstack([data["y_prod"][data["train_idx"]],
                            data["y_bench"][data["dev_idx"]]])
    weights_bq = np.ones(len(X_train_bq))
    weights_bq[-len(data["dev_idx"]):] = 100.0

    X_val = data["X_prod"][data["val_idx"]]
    y_val = data["y_prod"][data["val_idx"]]

    models_bq = train_xgboost_biquality(X_train_bq, y_train_bq, weights_bq,
                                          X_val, y_val, config, seed=seed)
    metrics_bq = evaluate(models_bq, X_bench_test, y_bench_test)
    logger.info("  Biquality (%d heuristic + %d GT): Micro-F1=%.4f, Macro-F1=%.4f",
                len(data["train_idx"]), len(data["dev_idx"]),
                metrics_bq["micro_f1"], metrics_bq["macro_f1"])
    results["biquality"] = {
        "n_heuristic": len(data["train_idx"]),
        "n_gt": len(data["dev_idx"]),
        **metrics_bq,
    }

    # --- GT-only ---
    logger.info("Training GT-only (%d samples)...", len(data["dev_idx"]))
    # Use benchmark dev for training, a portion for early-stopping eval
    # With only 187 samples, use the production val set for early stopping
    models_gt = train_xgboost_simple(X_bench_dev, y_bench_dev,
                                      X_val, y_val, config, seed=seed)
    metrics_gt = evaluate(models_gt, X_bench_test, y_bench_test)
    logger.info("  GT-only (%d samples): Micro-F1=%.4f, Macro-F1=%.4f",
                len(data["dev_idx"]), metrics_gt["micro_f1"], metrics_gt["macro_f1"])
    results["gt_only"] = {
        "n_samples": len(data["dev_idx"]),
        **metrics_gt,
    }

    delta = metrics_bq["micro_f1"] - metrics_gt["micro_f1"]
    logger.info("  Delta Micro-F1: %+.4f (heuristic labels %s)",
                delta, "help" if delta > 0 else "hurt or neutral")

    return results


def run_ablation3_lobo(config, seed=42):
    """Ablation 3: Leave-One-Benchmark-Out."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION 3: Leave-One-Benchmark-Out")
    logger.info("=" * 70)

    data = load_all_data(config, drop_derived=False)
    bench_labels = data["bench_labels"]

    X_prod_train = data["X_prod"][data["train_idx"]]
    y_prod_train = data["y_prod"][data["train_idx"]]
    X_val = data["X_prod"][data["val_idx"]]
    y_val = data["y_prod"][data["val_idx"]]

    benchmarks = bench_labels["benchmark"].unique()
    results = {}

    for bm in sorted(benchmarks):
        logger.info("")
        logger.info("--- Leave out: %s ---", bm)

        # Identify dev indices for this benchmark
        dev_mask_bm = bench_labels.iloc[data["dev_idx"]]["benchmark"].values == bm
        # Identify test indices for this benchmark
        test_mask_bm = bench_labels.iloc[data["test_idx"]]["benchmark"].values == bm

        # Dev set WITHOUT this benchmark
        dev_idx_without = data["dev_idx"][~dev_mask_bm]
        # Test set FOR this benchmark only
        test_idx_bm = data["test_idx"][test_mask_bm]

        if len(test_idx_bm) == 0:
            logger.info("  No test samples for %s, skipping", bm)
            continue

        n_dev_without = len(dev_idx_without)
        n_dev_removed = int(dev_mask_bm.sum())
        n_test = len(test_idx_bm)

        logger.info("  Dev: %d samples (removed %d %s samples), Test: %d %s samples",
                     n_dev_without, n_dev_removed, bm, n_test, bm)

        X_bench_dev_without = data["X_bench"][dev_idx_without]
        y_bench_dev_without = data["y_bench"][dev_idx_without]
        X_test_bm = data["X_bench"][test_idx_bm]
        y_test_bm = data["y_bench"][test_idx_bm]

        # --- With this benchmark (full biquality) ---
        X_train_full = np.vstack([X_prod_train, data["X_bench"][data["dev_idx"]]])
        y_train_full = np.vstack([y_prod_train, data["y_bench"][data["dev_idx"]]])
        w_full = np.ones(len(X_train_full))
        w_full[-len(data["dev_idx"]):] = 100.0

        models_full = train_xgboost_biquality(X_train_full, y_train_full, w_full,
                                               X_val, y_val, config, seed=seed)
        metrics_full = evaluate(models_full, X_test_bm, y_test_bm)

        # --- Without this benchmark ---
        if n_dev_without > 0:
            X_train_without = np.vstack([X_prod_train, X_bench_dev_without])
            y_train_without = np.vstack([y_prod_train, y_bench_dev_without])
            w_without = np.ones(len(X_train_without))
            w_without[-n_dev_without:] = 100.0
        else:
            # No GT dev data at all — heuristic only
            X_train_without = X_prod_train
            y_train_without = y_prod_train
            w_without = np.ones(len(X_train_without))

        models_without = train_xgboost_biquality(X_train_without, y_train_without,
                                                   w_without, X_val, y_val,
                                                   config, seed=seed)
        metrics_without = evaluate(models_without, X_test_bm, y_test_bm)

        delta = metrics_full["micro_f1"] - metrics_without["micro_f1"]
        logger.info("  With %s:    Micro-F1=%.4f, Macro-F1=%.4f",
                     bm, metrics_full["micro_f1"], metrics_full["macro_f1"])
        logger.info("  Without %s: Micro-F1=%.4f, Macro-F1=%.4f",
                     bm, metrics_without["micro_f1"], metrics_without["macro_f1"])
        logger.info("  Delta: %+.4f", delta)

        results[bm] = {
            "n_dev_with": len(data["dev_idx"]),
            "n_dev_without": n_dev_without,
            "n_dev_removed": n_dev_removed,
            "n_test": n_test,
            "with_benchmark": metrics_full,
            "without_benchmark": metrics_without,
            "delta_micro_f1": float(delta),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="ML Ablation Studies for IOSage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    seed = args.seed

    all_results = {}

    t0 = time.time()

    # Ablation 1
    all_results["ablation1_no_derived"] = run_ablation1_no_derived(config, seed=seed)

    # Ablation 2
    all_results["ablation2_gt_only"] = run_ablation2_gt_only(config, seed=seed)

    # Ablation 3
    all_results["ablation3_lobo"] = run_ablation3_lobo(config, seed=seed)

    total_time = time.time() - t0

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY OF ALL ABLATIONS")
    logger.info("=" * 70)

    a1 = all_results["ablation1_no_derived"]
    logger.info("")
    logger.info("Ablation 1: Derived Features")
    logger.info("  All features (%d): Micro-F1=%.4f",
                a1["all_features"]["n_features"], a1["all_features"]["micro_f1"])
    logger.info("  Raw only (%d):     Micro-F1=%.4f",
                a1["raw_only"]["n_features"], a1["raw_only"]["micro_f1"])
    logger.info("  Delta: %+.4f",
                a1["all_features"]["micro_f1"] - a1["raw_only"]["micro_f1"])

    a2 = all_results["ablation2_gt_only"]
    logger.info("")
    logger.info("Ablation 2: GT-Only vs Biquality")
    logger.info("  Biquality (%d+%d): Micro-F1=%.4f",
                a2["biquality"]["n_heuristic"], a2["biquality"]["n_gt"],
                a2["biquality"]["micro_f1"])
    logger.info("  GT-only (%d):      Micro-F1=%.4f",
                a2["gt_only"]["n_samples"], a2["gt_only"]["micro_f1"])
    logger.info("  Delta: %+.4f",
                a2["biquality"]["micro_f1"] - a2["gt_only"]["micro_f1"])

    a3 = all_results["ablation3_lobo"]
    logger.info("")
    logger.info("Ablation 3: Leave-One-Benchmark-Out")
    logger.info("  %-12s %10s %10s %8s", "Benchmark", "With", "Without", "Delta")
    for bm in sorted(a3.keys()):
        r = a3[bm]
        logger.info("  %-12s %10.4f %10.4f %+8.4f",
                     bm,
                     r["with_benchmark"]["micro_f1"],
                     r["without_benchmark"]["micro_f1"],
                     r["delta_micro_f1"])

    logger.info("")
    logger.info("Total time: %.1f seconds", total_time)

    # Save results
    out_path = PROJECT_DIR / "results" / "ml_ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
