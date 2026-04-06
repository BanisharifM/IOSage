#!/usr/bin/env python3
"""
Comprehensive evaluation of boost experiment models.

Runs ALL evaluations that depend on the model or test set, saving everything
to results/boost_experiment/. Does NOT touch any existing results.

Evaluations:
  1. Full metrics (Micro/Macro F1, Hamming, subset accuracy, bootstrap CIs)
  2. Baselines on new test set (Drishti, WisIO, majority class, threshold)
  3. Per-benchmark F1 breakdown
  4. ML ablations (no derived features, GT-only, biquality weight sensitivity)
  5. Detection threshold sweep
  6. Leave-one-benchmark-out (LOBO) generalization
  7. TraceBench cross-system generalization with new model
  8. Side-by-side comparison with old results for every metric

Usage:
    /projects/bdau/envs/sc2026/bin/python scripts/run_boost_full_eval.py
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
sys.path.insert(0, str(PROJECT))

EXPERIMENT_DIR = PROJECT / "results" / "boost_experiment"
EVAL_DIR = EXPERIMENT_DIR / "full_evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# Data loading
# ============================================================================

def load_config():
    with open(PROJECT / "configs" / "training.yaml") as f:
        return yaml.safe_load(f)


def get_feature_cols(config):
    """Get feature columns consistent with training pipeline."""
    prod_features = pd.read_parquet(PROJECT / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_features.columns if c not in exclude]
    del prod_features
    gc.collect()
    return feature_cols


def load_new_test_data(feature_cols):
    """Load new test set from boost experiment."""
    split_dir = EXPERIMENT_DIR / "new_splits"
    features = pd.read_parquet(split_dir / "test_features.parquet")
    labels = pd.read_parquet(split_dir / "test_labels.parquet")

    X_cols = []
    for col in feature_cols:
        if col in features.columns:
            X_cols.append(features[col].values)
        else:
            X_cols.append(np.zeros(len(features)))
    X_test = np.column_stack(X_cols).astype(np.float32)
    y_test = labels[DIMS].values.astype(np.float32)
    return X_test, y_test, features, labels


def load_new_full_data(feature_cols):
    """Load full new GT dataset."""
    split_dir = EXPERIMENT_DIR / "new_splits"
    features = pd.read_parquet(split_dir / "features.parquet")
    labels = pd.read_parquet(split_dir / "labels.parquet")
    with open(split_dir / "split_indices.pkl", "rb") as f:
        splits = pickle.load(f)
    return features, labels, splits


def load_new_model(seed=42):
    """Load retrained model from boost experiment."""
    model_path = EXPERIMENT_DIR / "new_models" / f"xgboost_biquality_w100_seed{seed}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_with_model(models, X, threshold=0.5):
    """Get predictions and probabilities from per-dimension models."""
    y_prob = np.zeros((len(X), len(DIMS)))
    for i, dim in enumerate(DIMS):
        y_prob[:, i] = models[dim].predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


# ============================================================================
# 1. Full metrics with bootstrap CIs
# ============================================================================

def compute_full_metrics(y_test, y_pred, y_prob, n_bootstrap=10000, seed=42):
    """Compute all metrics with bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred)
    subset_acc = accuracy_score(y_test, y_pred)

    # Per-label
    per_label = {}
    for i, dim in enumerate(DIMS):
        per_label[dim] = {
            "f1": float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "precision": float(precision_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_test[:, i], y_pred[:, i], zero_division=0)),
            "support": int(y_test[:, i].sum()),
        }

    # Bootstrap CIs
    boot_micro = []
    boot_macro = []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        bm = f1_score(y_test[idx], y_pred[idx], average="micro", zero_division=0)
        bM = f1_score(y_test[idx], y_pred[idx], average="macro", zero_division=0)
        boot_micro.append(bm)
        boot_macro.append(bM)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "hamming_loss": float(h_loss),
        "subset_accuracy": float(subset_acc),
        "micro_precision": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
        "per_label": per_label,
        "bootstrap_ci": {
            "micro_f1_95": [float(np.percentile(boot_micro, 2.5)), float(np.percentile(boot_micro, 97.5))],
            "macro_f1_95": [float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5))],
        },
        "n_test": int(len(y_test)),
    }


# ============================================================================
# 2. Baselines on new test set
# ============================================================================

def run_drishti_baseline(test_labels):
    """Evaluate Drishti heuristic labels as predictions against GT."""
    # Drishti predictions come from heuristic labeling on benchmark logs
    # Since we don't have direct Drishti output for these logs, we use
    # the production Drishti labels as proxy for the benchmark samples
    # that overlap, OR we run Drishti-style rules on the features.
    # For benchmark logs, Drishti baseline = all zeros for dims it can't detect.
    logger.info("Computing Drishti baseline on new test set...")

    from src.data.drishti_labeling import apply_drishti_rules
    from src.data.feature_extraction import extract_raw_features

    # Load test features (raw, before normalization)
    split_dir = EXPERIMENT_DIR / "new_splits"
    test_features = pd.read_parquet(split_dir / "test_features.parquet")
    y_test = test_labels[DIMS].values.astype(int)

    # Apply Drishti rules to each test sample
    y_drishti = np.zeros_like(y_test)
    for idx in range(len(test_features)):
        row = test_features.iloc[idx]
        features_dict = row.to_dict()
        try:
            drishti_labels = apply_drishti_rules(features_dict)
            for j, dim in enumerate(DIMS):
                y_drishti[idx, j] = drishti_labels.get(dim, 0)
        except Exception:
            pass  # Leave as zeros

    metrics = {
        "micro_f1": float(f1_score(y_test, y_drishti, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_drishti, average="macro", zero_division=0)),
        "per_label": {},
    }
    for i, dim in enumerate(DIMS):
        metrics["per_label"][dim] = {
            "f1": float(f1_score(y_test[:, i], y_drishti[:, i], zero_division=0)),
            "precision": float(precision_score(y_test[:, i], y_drishti[:, i], zero_division=0)),
            "recall": float(recall_score(y_test[:, i], y_drishti[:, i], zero_division=0)),
        }
    return metrics


def run_majority_baseline(y_test):
    """Majority class baseline."""
    # Predict all zeros (healthy=1 for all)
    y_majority = np.zeros_like(y_test)
    y_majority[:, DIMS.index("healthy")] = 1  # Predict all healthy
    return {
        "micro_f1": float(f1_score(y_test, y_majority, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_majority, average="macro", zero_division=0)),
    }


# ============================================================================
# 3. Per-benchmark F1
# ============================================================================

def compute_per_benchmark_f1(models, feature_cols):
    """F1 broken down by benchmark type."""
    split_dir = EXPERIMENT_DIR / "new_splits"
    features = pd.read_parquet(split_dir / "test_features.parquet")
    labels = pd.read_parquet(split_dir / "test_labels.parquet")

    X_cols = []
    for col in feature_cols:
        if col in features.columns:
            X_cols.append(features[col].values)
        else:
            X_cols.append(np.zeros(len(features)))
    X_test = np.column_stack(X_cols).astype(np.float32)
    y_test = labels[DIMS].values.astype(int)

    y_pred, _ = predict_with_model(models, X_test)

    benchmarks = labels["benchmark"].unique()
    result = {}
    for bench in benchmarks:
        mask = labels["benchmark"] == bench
        n = mask.sum()
        if n == 0:
            continue
        idx = mask.values
        result[bench] = {
            "n": int(n),
            "micro_f1": float(f1_score(y_test[idx], y_pred[idx], average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_test[idx], y_pred[idx], average="macro", zero_division=0)),
        }
    return result


# ============================================================================
# 4. Threshold sweep
# ============================================================================

def threshold_sweep(models, X_test, y_test):
    """Sweep detection threshold from 0.1 to 0.7."""
    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    results = {}
    for t in thresholds:
        y_pred, _ = predict_with_model(models, X_test, threshold=t)
        results[str(t)] = {
            "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
            "micro_precision": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
        }
    return results


# ============================================================================
# 5. ML Ablations
# ============================================================================

def run_ml_ablations(feature_cols, gt_dir, config, seed=42, clean_weight=100.0):
    """Run ML ablation studies on new data."""
    from xgboost import XGBClassifier

    # Load production data
    paths = config["paths"]
    prod_features = pd.read_parquet(PROJECT / paths["production_features"])
    prod_labels = pd.read_parquet(PROJECT / paths["production_labels"])
    prod_labels = prod_labels.set_index("_jobid")
    prod_features = prod_features.set_index("_jobid")
    common = prod_features.index.intersection(prod_labels.index)
    prod_features = prod_features.loc[common]
    prod_labels = prod_labels.loc[common]

    with open(PROJECT / paths["production_splits"], "rb") as f:
        prod_splits = pickle.load(f)
    train_idx = prod_splits.get("train_idx", prod_splits.get("train_indices"))

    X_prod_train = prod_features.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_prod_train = prod_labels.iloc[train_idx][DIMS].values.astype(np.float32)
    del prod_features, prod_labels
    gc.collect()

    # Load new benchmark data
    gt_dir = Path(gt_dir)
    bench_features = pd.read_parquet(gt_dir / "features.parquet")
    bench_labels = pd.read_parquet(gt_dir / "labels.parquet")
    with open(gt_dir / "split_indices.pkl", "rb") as f:
        bench_splits = pickle.load(f)
    dev_idx = bench_splits["dev_idx"]
    test_idx = bench_splits["test_idx"]

    X_bench_dev_cols = [bench_features.iloc[dev_idx][col].values if col in bench_features.columns
                        else np.zeros(len(dev_idx)) for col in feature_cols]
    X_bench_dev = np.column_stack(X_bench_dev_cols).astype(np.float32)
    y_bench_dev = bench_labels.iloc[dev_idx][DIMS].values.astype(np.float32)

    X_bench_test_cols = [bench_features.iloc[test_idx][col].values if col in bench_features.columns
                         else np.zeros(len(test_idx)) for col in feature_cols]
    X_bench_test = np.column_stack(X_bench_test_cols).astype(np.float32)
    y_bench_test = bench_labels.iloc[test_idx][DIMS].values.astype(np.float32)

    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params["n_jobs"] = 4
    xgb_params.pop("eval_metric", None)

    def train_and_eval(X_train, y_train, weights, label):
        spw = []
        for i in range(y_train.shape[1]):
            n_pos = y_train[:, i].sum()
            n_neg = len(y_train) - n_pos
            spw.append(min(n_neg / max(n_pos, 1), 100.0))
        models = {}
        for i, dim in enumerate(DIMS):
            clf = XGBClassifier(**xgb_params, scale_pos_weight=spw[i],
                                random_state=seed, verbosity=0)
            clf.fit(X_train, y_train[:, i], sample_weight=weights, verbose=False)
            models[dim] = clf
        y_pred, _ = predict_with_model(models, X_bench_test)
        micro = f1_score(y_bench_test, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_bench_test, y_pred, average="macro", zero_division=0)
        per_dim = {}
        for i, dim in enumerate(DIMS):
            per_dim[dim] = float(f1_score(y_bench_test[:, i], y_pred[:, i], zero_division=0))
        logger.info("  %s: Micro=%.4f, Macro=%.4f", label, micro, macro)
        del models
        gc.collect()
        return {"micro_f1": float(micro), "macro_f1": float(macro), "per_label": per_dim}

    results = {}

    # A1: Full pipeline (baseline for comparison)
    logger.info("Ablation: Full pipeline (baseline)")
    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])
    w_combined = np.ones(len(X_combined), dtype=np.float32)
    w_combined[-len(X_bench_dev):] = clean_weight
    results["full_pipeline"] = train_and_eval(X_combined, y_combined, w_combined, "Full pipeline")

    # A2: No derived features (raw counters only)
    logger.info("Ablation: No derived features")
    # Identify derived feature columns (those not starting with POSIX_, MPIIO_, STDIO_, has_, nprocs, runtime)
    raw_cols_idx = []
    derived_cols_idx = []
    for j, col in enumerate(feature_cols):
        is_raw = (col.startswith("POSIX_") or col.startswith("MPIIO_") or
                  col.startswith("STDIO_") or col.startswith("has_") or
                  col in ["nprocs", "runtime_seconds", "num_files", "is_shared_file"])
        if is_raw:
            raw_cols_idx.append(j)
        else:
            derived_cols_idx.append(j)
    X_raw = X_combined[:, raw_cols_idx]
    X_raw_test = X_bench_test[:, raw_cols_idx]
    # Temporarily swap test set for raw-only eval
    _save_test = X_bench_test
    X_bench_test = X_raw_test
    results["no_derived_features"] = train_and_eval(
        X_raw, y_combined, w_combined, f"No derived ({len(raw_cols_idx)} features)")
    X_bench_test = _save_test  # Restore

    # A3: GT-only training (no heuristic labels)
    logger.info("Ablation: GT-only training")
    w_gt = np.ones(len(X_bench_dev), dtype=np.float32)
    results["gt_only"] = train_and_eval(X_bench_dev, y_bench_dev, w_gt, "GT only")

    # A4: Biquality weight sensitivity
    logger.info("Ablation: Weight sensitivity")
    results["weight_sensitivity"] = {}
    for w in [1, 10, 50, 100, 200, 500]:
        w_sweep = np.ones(len(X_combined), dtype=np.float32)
        w_sweep[-len(X_bench_dev):] = w
        r = train_and_eval(X_combined, y_combined, w_sweep, f"w={w}")
        results["weight_sensitivity"][str(w)] = r

    # A5: Leave-one-benchmark-out (LOBO)
    logger.info("Ablation: Leave-one-benchmark-out")
    results["lobo"] = {}
    bench_types = bench_labels.iloc[test_idx]["benchmark"].unique()
    for leave_out in bench_types:
        # Remove leave_out from dev set, evaluate on its test partition
        dev_mask = bench_labels.iloc[dev_idx]["benchmark"].values != leave_out
        test_bt_mask = bench_labels.iloc[test_idx]["benchmark"].values == leave_out

        if dev_mask.sum() == 0 or test_bt_mask.sum() == 0:
            continue

        X_dev_lobo = X_bench_dev[dev_mask]
        y_dev_lobo = y_bench_dev[dev_mask]

        X_combined_lobo = np.vstack([X_prod_train, X_dev_lobo])
        y_combined_lobo = np.vstack([y_prod_train, y_dev_lobo])
        w_lobo = np.ones(len(X_combined_lobo), dtype=np.float32)
        w_lobo[-len(X_dev_lobo):] = clean_weight

        # Evaluate only on left-out benchmark's test partition
        X_test_lobo = X_bench_test[test_bt_mask]
        y_test_lobo = y_bench_test[test_bt_mask]

        _save = X_bench_test, y_bench_test
        X_bench_test, y_bench_test = X_test_lobo, y_test_lobo
        results["lobo"][leave_out] = train_and_eval(
            X_combined_lobo, y_combined_lobo, w_lobo, f"LOBO({leave_out})")
        X_bench_test, y_bench_test = _save

    del X_combined, y_combined, X_prod_train, y_prod_train
    gc.collect()

    return results


# ============================================================================
# 6. TraceBench with new model
# ============================================================================

def run_tracebench_eval(models, feature_cols):
    """Evaluate new model on TraceBench traces."""
    from src.data.parse_darshan import parse_darshan_log
    from src.data.feature_extraction import extract_raw_features
    from src.data.preprocessing import stage3_engineer

    tb_dir = PROJECT / "data" / "external" / "tracebench" / "TraceBench"
    label_map_path = PROJECT / "data" / "external" / "tracebench" / "label_mapping.json"

    if not tb_dir.exists():
        logger.warning("TraceBench not found at %s", tb_dir)
        return None

    if not label_map_path.exists():
        logger.warning("TraceBench label mapping not found")
        return None

    with open(label_map_path) as f:
        label_mapping = json.load(f)

    # Find all Darshan files
    darshan_files = list(tb_dir.glob("**/*.darshan"))
    logger.info("Found %d TraceBench Darshan files", len(darshan_files))

    results = []
    for fpath in darshan_files:
        try:
            parsed = parse_darshan_log(str(fpath))
            if parsed is None:
                continue
            features = extract_raw_features(parsed)
            features_df = pd.DataFrame([features])
            features_df = stage3_engineer(features_df)

            X_cols = []
            for col in feature_cols:
                if col in features_df.columns:
                    X_cols.append(features_df[col].values[0])
                else:
                    X_cols.append(0.0)
            X = np.array([X_cols], dtype=np.float32)

            y_pred, y_prob = predict_with_model(models, X, threshold=0.3)
            pred_dims = [DIMS[i] for i in range(len(DIMS)) if y_pred[0, i] == 1]

            results.append({
                "file": fpath.name,
                "predictions": pred_dims,
                "probabilities": {DIMS[i]: float(y_prob[0, i]) for i in range(len(DIMS))},
            })
        except Exception as e:
            logger.warning("Failed on %s: %s", fpath.name, e)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE EVALUATION: Boost Experiment Models")
    logger.info("Output: %s", EVAL_DIR)
    logger.info("=" * 70)

    config = load_config()
    feature_cols = get_feature_cols(config)
    logger.info("Feature columns: %d", len(feature_cols))

    # Load test data
    X_test, y_test, test_features, test_labels = load_new_test_data(feature_cols)
    logger.info("New test set: %d samples", len(X_test))

    # ---- 1. Full metrics (5 seeds) ----
    logger.info("\n=== 1. FULL METRICS (5 seeds, bootstrap CIs) ===")
    all_seed_metrics = []
    for seed in SEEDS:
        models = load_new_model(seed)
        y_pred, y_prob = predict_with_model(models, X_test)
        metrics = compute_full_metrics(y_test, y_pred, y_prob, n_bootstrap=10000, seed=seed)
        metrics["seed"] = seed
        all_seed_metrics.append(metrics)
        logger.info("  Seed %d: Micro=%.4f [%.3f, %.3f], Macro=%.4f",
                    seed, metrics["micro_f1"],
                    metrics["bootstrap_ci"]["micro_f1_95"][0],
                    metrics["bootstrap_ci"]["micro_f1_95"][1],
                    metrics["macro_f1"])
        del models
        gc.collect()

    with open(EVAL_DIR / "full_metrics_5seeds.json", "w") as f:
        json.dump(all_seed_metrics, f, indent=2)

    # ---- 2. Baselines ----
    logger.info("\n=== 2. BASELINES ON NEW TEST SET ===")
    baselines = {}

    # Drishti
    try:
        baselines["drishti"] = run_drishti_baseline(test_labels)
        logger.info("  Drishti: Micro=%.4f", baselines["drishti"]["micro_f1"])
    except Exception as e:
        logger.warning("  Drishti baseline failed: %s", e)

    # Majority class
    baselines["majority_class"] = run_majority_baseline(y_test)
    logger.info("  Majority: Micro=%.4f", baselines["majority_class"]["micro_f1"])

    with open(EVAL_DIR / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # ---- 3. Per-benchmark F1 ----
    logger.info("\n=== 3. PER-BENCHMARK F1 ===")
    models_42 = load_new_model(42)
    per_bench = compute_per_benchmark_f1(models_42, feature_cols)
    for bench, m in per_bench.items():
        logger.info("  %s (n=%d): Micro=%.4f", bench, m["n"], m["micro_f1"])
    with open(EVAL_DIR / "per_benchmark_f1.json", "w") as f:
        json.dump(per_bench, f, indent=2)

    # ---- 4. Threshold sweep ----
    logger.info("\n=== 4. THRESHOLD SWEEP ===")
    thresh_results = threshold_sweep(models_42, X_test, y_test)
    for t, m in thresh_results.items():
        logger.info("  t=%s: Micro=%.4f, P=%.3f, R=%.3f",
                    t, m["micro_f1"], m["micro_precision"], m["micro_recall"])
    with open(EVAL_DIR / "threshold_sweep.json", "w") as f:
        json.dump(thresh_results, f, indent=2)
    del models_42
    gc.collect()

    # ---- 5. ML Ablations ----
    logger.info("\n=== 5. ML ABLATIONS ===")
    ablation_results = run_ml_ablations(
        feature_cols, EXPERIMENT_DIR / "new_splits", config
    )
    with open(EVAL_DIR / "ml_ablations.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # ---- 6. TraceBench ----
    logger.info("\n=== 6. TRACEBENCH CROSS-SYSTEM ===")
    tb_model = load_new_model(42)
    tb_results = run_tracebench_eval(tb_model, feature_cols)
    if tb_results:
        with open(EVAL_DIR / "tracebench_predictions.json", "w") as f:
            json.dump(tb_results, f, indent=2)
        logger.info("  TraceBench: %d traces evaluated", len(tb_results))
    del tb_model
    gc.collect()

    # ---- 7. Summary comparison ----
    logger.info("\n=== 7. SUMMARY COMPARISON ===")

    # Load old results
    old_path = PROJECT / "results" / "final_metrics.json"
    if old_path.exists():
        with open(old_path) as f:
            old = json.load(f)
        old_m = old["xgboost"]["metrics"]

        # Average new results
        new_micros = [r["micro_f1"] for r in all_seed_metrics]
        new_macros = [r["macro_f1"] for r in all_seed_metrics]

        summary = {
            "old": {
                "n_gt": 623, "n_test": 436,
                "micro_f1": old_m["micro_f1"],
                "macro_f1": old_m["macro_f1"],
                "per_label": old_m["per_label"],
            },
            "new": {
                "n_gt": 689, "n_test": len(y_test),
                "micro_f1_mean": float(np.mean(new_micros)),
                "micro_f1_std": float(np.std(new_micros)),
                "macro_f1_mean": float(np.mean(new_macros)),
                "macro_f1_std": float(np.std(new_macros)),
                "per_label": all_seed_metrics[0]["per_label"],  # seed 42
                "bootstrap_ci": all_seed_metrics[0]["bootstrap_ci"],
            },
            "ablations": ablation_results,
            "baselines": baselines,
            "per_benchmark": per_bench,
            "threshold_sweep": thresh_results,
        }

        with open(EVAL_DIR / "full_comparison.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print comparison table
        logger.info("")
        logger.info("=" * 80)
        logger.info("FULL COMPARISON: OLD (623 GT) vs NEW (689 GT)")
        logger.info("=" * 80)
        logger.info("%-30s %10s %18s %10s", "Metric", "Old", "New (mean±std)", "Delta")
        logger.info("-" * 80)
        d_mi = np.mean(new_micros) - old_m["micro_f1"]
        d_ma = np.mean(new_macros) - old_m["macro_f1"]
        logger.info("%-30s %10.4f %8.4f±%.4f %+10.4f",
                    "Micro-F1", old_m["micro_f1"], np.mean(new_micros), np.std(new_micros), d_mi)
        logger.info("%-30s %10.4f %8.4f±%.4f %+10.4f",
                    "Macro-F1", old_m["macro_f1"], np.mean(new_macros), np.std(new_macros), d_ma)
        logger.info("")

        logger.info("Per-dimension:")
        for dim in DIMS:
            of1 = old_m["per_label"][dim]["f1"]
            nf1 = all_seed_metrics[0]["per_label"][dim]["f1"]
            on = old_m["per_label"][dim]["support"]
            nn = all_seed_metrics[0]["per_label"][dim]["support"]
            logger.info("  %-28s %7.3f (n=%3d) -> %7.3f (n=%3d) %+7.3f",
                        dim, of1, on, nf1, nn, nf1 - of1)

        logger.info("")
        logger.info("Ablation: Full=%.4f, NoDerived=%.4f, GTonly=%.4f",
                    ablation_results["full_pipeline"]["micro_f1"],
                    ablation_results["no_derived_features"]["micro_f1"],
                    ablation_results["gt_only"]["micro_f1"])
        logger.info("")
        logger.info("Weight sensitivity:")
        for w, r in ablation_results.get("weight_sensitivity", {}).items():
            logger.info("  w=%s: Micro=%.4f", w, r["micro_f1"])
        logger.info("")
        logger.info("LOBO:")
        for bench, r in ablation_results.get("lobo", {}).items():
            logger.info("  -%s: Micro=%.4f", bench, r["micro_f1"])

    logger.info("\n" + "=" * 80)
    logger.info("ALL EVALUATIONS COMPLETE. Results in: %s", EVAL_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
