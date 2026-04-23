"""
Standalone evaluation module for multi-label I/O bottleneck classifiers.

Computes all metrics required for IOSage evaluation:
- Primary: Micro-F1, Macro-F1
- Secondary: Hamming loss, per-label F1/P/R, subset accuracy
- Statistical: Bootstrap 95% CI, paired model comparison
- Visualization: Confusion matrices, classification reports

Can evaluate:
- Trained ML models (XGBoost, LightGBM, RF, MLP)
- Drishti baseline (vectorized heuristic rules)
- Threshold baselines
- Majority class baseline
- Logistic regression baseline

Usage:
    python -m src.models.evaluate --models models/xgboost_br_models.pkl
    python -m src.models.evaluate --baseline drishti
    python -m src.models.evaluate --baseline majority
"""

import argparse
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSION_NAMES = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
    "healthy",
]


def compute_all_metrics(y_true, y_pred, dim_names=None):
    """Compute comprehensive multi-label metrics.

    Returns dict with all metrics needed for SC paper.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        hamming_loss, accuracy_score,
    )

    if dim_names is None:
        dim_names = DIMENSION_NAMES

    metrics = {}

    # Overall metrics
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["subset_accuracy"] = accuracy_score(
        y_true.astype(int).tolist(),
        y_pred.astype(int).tolist(),
    ) if y_true.ndim == 2 else 0.0

    # Per-label metrics
    metrics["per_label"] = {}
    for i, dim in enumerate(dim_names):
        yt = y_true[:, i] if y_true.ndim == 2 else y_true
        yp = y_pred[:, i] if y_pred.ndim == 2 else y_pred
        n_pos = yt.sum()
        metrics["per_label"][dim] = {
            "f1": f1_score(yt, yp, zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "support": int(n_pos),
        }

    return metrics


def bootstrap_ci(y_true, y_pred, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap confidence intervals for Micro-F1 and Macro-F1."""
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    micro_scores, macro_scores = [], []

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        micro_scores.append(f1_score(y_true[idx], y_pred[idx], average="micro", zero_division=0))
        macro_scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    alpha = (1 - confidence) / 2
    return {
        "micro_f1_mean": np.mean(micro_scores),
        "micro_f1_lo": np.percentile(micro_scores, alpha * 100),
        "micro_f1_hi": np.percentile(micro_scores, (1 - alpha) * 100),
        "macro_f1_mean": np.mean(macro_scores),
        "macro_f1_lo": np.percentile(macro_scores, alpha * 100),
        "macro_f1_hi": np.percentile(macro_scores, (1 - alpha) * 100),
    }


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, n_resamples=10000, seed=42):
    """Test if model A significantly outperforms model B (paired bootstrap)."""
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        f1_a = f1_score(y_true[idx], y_pred_a[idx], average="micro", zero_division=0)
        f1_b = f1_score(y_true[idx], y_pred_b[idx], average="micro", zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    p_value = np.mean(diffs <= 0)  # fraction where B >= A
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    return {
        "mean_diff": np.mean(diffs),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "significant": ci_lo > 0,  # True if A significantly better
    }


def compute_confusion_matrices(y_true, y_pred, dim_names=None):
    """Compute per-label 2x2 confusion matrices."""
    from sklearn.metrics import confusion_matrix

    if dim_names is None:
        dim_names = DIMENSION_NAMES

    cms = {}
    for i, dim in enumerate(dim_names):
        yt = y_true[:, i] if y_true.ndim == 2 else y_true
        yp = y_pred[:, i] if y_pred.ndim == 2 else y_pred
        if yt.sum() > 0 or yp.sum() > 0:
            cms[dim] = confusion_matrix(yt, yp, labels=[0, 1])
        else:
            cms[dim] = np.array([[len(yt), 0], [0, 0]])
    return cms


def predict_with_models(models, X, dim_names=None):
    """Predict using a dict of per-label models."""
    if dim_names is None:
        dim_names = DIMENSION_NAMES
    y_pred = np.zeros((len(X), len(dim_names)))
    for i, dim in enumerate(dim_names):
        if dim in models:
            y_pred[:, i] = models[dim].predict(X)
    return y_pred


def predict_drishti_baseline(X_features_df, dim_names=None):
    """Apply Drishti heuristic rules to features and return predictions.

    Uses the same vectorized Drishti implementation as heuristic labeling.
    """
    sys.path.insert(0, str(PROJECT_DIR))
    from src.data.drishti_labeling import apply_drishti_labels

    if dim_names is None:
        dim_names = DIMENSION_NAMES

    labels_df = apply_drishti_labels(X_features_df)
    y_pred = labels_df[dim_names].values.astype(np.float32)
    return y_pred


def predict_majority_baseline(y_train, n_test, dim_names=None):
    """Predict the majority class for each label (based on training distribution)."""
    if dim_names is None:
        dim_names = DIMENSION_NAMES
    y_pred = np.zeros((n_test, len(dim_names)))
    for i in range(y_train.shape[1]):
        majority = 1 if y_train[:, i].mean() > 0.5 else 0
        y_pred[:, i] = majority
    return y_pred


def predict_threshold_baseline(X, feature_cols, dim_names=None, percentile=90):
    """Threshold-based baseline: flag if any relevant feature > Pth percentile."""
    if dim_names is None:
        dim_names = DIMENSION_NAMES

    # Simple: predict 1 if any feature is above the percentile threshold
    thresholds = np.percentile(X, percentile, axis=0)
    y_pred = np.zeros((len(X), len(dim_names)))

    # Map features to dimensions (simplified — uses feature name patterns)
    dim_feature_map = {
        "access_granularity": ["small_io_ratio", "small_read_ratio", "small_write_ratio"],
        "metadata_intensity": ["metadata_time_ratio", "opens_per_op", "stats_per_op"],
        "access_pattern": ["seq_read_ratio", "seq_write_ratio"],  # inverted: low seq = random
        "interface_choice": ["collective_ratio"],  # inverted: low collective = bad
        "throughput_utilization": ["fsync_ratio", "total_bw_mb_s"],  # inverted for bw
        "healthy": [],
    }

    for i, dim in enumerate(dim_names):
        feat_names = dim_feature_map.get(dim, [])
        for fname in feat_names:
            if fname in feature_cols:
                idx = feature_cols.index(fname)
                if dim == "access_pattern":
                    # Low sequential = random access
                    y_pred[:, i] |= (X[:, idx] < np.percentile(X[:, idx], 100 - percentile))
                elif dim == "interface_choice":
                    y_pred[:, i] |= (X[:, idx] < np.percentile(X[:, idx], 100 - percentile))
                else:
                    y_pred[:, i] |= (X[:, idx] > thresholds[idx])

    return y_pred


def log_evaluation_report(name, metrics, ci=None):
    """Print formatted evaluation report."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation: %s", name)
    logger.info("=" * 70)
    logger.info("Micro-F1: %.4f  Macro-F1: %.4f  Hamming: %.4f  SubsetAcc: %.4f",
                metrics["micro_f1"], metrics["macro_f1"],
                metrics["hamming_loss"], metrics.get("subset_accuracy", 0))

    if ci:
        logger.info("Bootstrap 95%% CI: Micro-F1=[%.4f, %.4f]  Macro-F1=[%.4f, %.4f]",
                    ci["micro_f1_lo"], ci["micro_f1_hi"],
                    ci["macro_f1_lo"], ci["macro_f1_hi"])

    logger.info("")
    header = f"{'Dimension':<28s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Supp':>6s}"
    logger.info(header)
    logger.info("-" * len(header))
    for dim, m in metrics["per_label"].items():
        logger.info(f"{dim:<28s} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['support']:6d}")
    logger.info("=" * 70)


def measure_inference_latency(models, X_sample, n_warmup=10, n_runs=100):
    """Measure per-sample inference latency (ms)."""
    # Warmup
    for _ in range(n_warmup):
        predict_with_models(models, X_sample[:1])

    # Measure
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_with_models(models, X_sample[:1])
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }
