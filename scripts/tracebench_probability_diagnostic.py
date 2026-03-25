#!/usr/bin/env python
"""
TraceBench Probability Diagnostic
===================================
Determines whether our XGBoost model's predicted probabilities contain useful
signal for TraceBench, even though binary predictions at threshold=0.5 yield
poor F1 (0.103).

Computes per-dimension AUC-ROC, AUC-PR, probability distributions for true
positives vs true negatives, and optimal thresholds via LOO-CV.
"""

import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_extraction import extract_raw_features, FEATURE_GROUPS
from src.data.preprocessing import stage3_engineer, stage5_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (reuse from evaluate_tracebench.py)
# ---------------------------------------------------------------------------
TRACEBENCH_ROOT = PROJECT_ROOT / "data" / "external" / "tracebench" / "TraceBench"
LABEL_MAPPING_PATH = PROJECT_ROOT / "data" / "external" / "tracebench" / "label_mapping.json"
MODEL_PATH = PROJECT_ROOT / "models" / "phase2" / "xgboost_biquality_w100.pkl"
SCALERS_PATH = PROJECT_ROOT / "data" / "processed" / "production" / "scalers.pkl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracebench_evaluation"

OUR_DIMENSIONS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
    "healthy",
]

TRACEBENCH_COVERED_DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
]

DROP_FEATURES = [
    "POSIX_F_OPEN_START_TIMESTAMP",
    "POSIX_F_READ_START_TIMESTAMP",
    "POSIX_F_WRITE_START_TIMESTAMP",
    "POSIX_F_CLOSE_START_TIMESTAMP",
    "POSIX_F_OPEN_END_TIMESTAMP",
    "POSIX_F_READ_END_TIMESTAMP",
    "POSIX_F_WRITE_END_TIMESTAMP",
    "POSIX_F_CLOSE_END_TIMESTAMP",
    "POSIX_RENAMED_FROM",
    "POSIX_MODE",
    "POSIX_FILE_ALIGNMENT",
    "POSIX_MEM_ALIGNMENT",
    "POSIX_MMAPS",
    "POSIX_FDSYNCS",
    "has_posix",
    "has_hdf5",
    "has_pnetcdf",
    "has_heatmap",
    "POSIX_F_VARIANCE_RANK_TIME",
    "POSIX_F_VARIANCE_RANK_BYTES",
    "MPIIO_F_VARIANCE_RANK_TIME",
    "MPIIO_F_VARIANCE_RANK_BYTES",
    "STDIO_F_VARIANCE_RANK_TIME",
    "STDIO_F_VARIANCE_RANK_BYTES",
    "MPIIO_SPLIT_READS",
    "MPIIO_SPLIT_WRITES",
    "MPIIO_NB_READS",
    "rank_bytes_cv",
    "rank_time_cv",
]


def parse_darshan_subprocess(darshan_path, timeout=60):
    """Parse a darshan file in a subprocess to avoid double-free crashes."""
    script = f"""
import sys, json
sys.path.insert(0, "{PROJECT_ROOT}")
from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features

parsed = parse_darshan_log("{darshan_path}", backend="cli")
if parsed is None:
    print("PARSE_FAILED")
    sys.exit(0)

features = extract_raw_features(parsed)
for k, v in features.items():
    if hasattr(v, 'item'):
        features[k] = v.item()
print(json.dumps(features))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("Subprocess error for %s: %s", darshan_path, result.stderr[:200])
            return None
        output = result.stdout.strip()
        if output == "PARSE_FAILED" or not output:
            return None
        return json.loads(output)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("Failed to parse %s: %s", darshan_path, e)
        return None


def load_tracebench_labels():
    """Load all TraceBench labels from the three subsets."""
    labels = {}
    for subset in ["single_issue_bench", "IO500", "real_app_bench"]:
        label_file = TRACEBENCH_ROOT / "Datasets" / subset / "trace_labels.json"
        if not label_file.exists():
            logger.warning("No label file for %s", subset)
            continue
        with open(label_file) as f:
            subset_labels = json.load(f)
        for trace_name, trace_labels in subset_labels.items():
            labels[(subset, trace_name)] = trace_labels
    return labels


def find_darshan_files(all_labels):
    """Find darshan files matched to trace names."""
    files = {}
    for subset in ["single_issue_bench", "IO500", "real_app_bench"]:
        darshan_dir = TRACEBENCH_ROOT / "Datasets" / subset / "darshan_files" / "darshan"
        if not darshan_dir.exists():
            logger.warning("No darshan dir for %s", subset)
            continue
        subset_keys = [k for k in all_labels if k[0] == subset]
        for f in sorted(darshan_dir.glob("*.darshan")):
            fname = f.stem
            if fname.endswith(".ini"):
                fname = fname[:-4]
            fname_norm = fname.replace(" ", "_")
            for key in subset_keys:
                key_norm = key[1].replace(" ", "_")
                if fname_norm == key_norm:
                    files[key] = f
                    break
            else:
                best_match = None
                best_len = 0
                for key in subset_keys:
                    key_norm = key[1].replace(" ", "_")
                    if fname_norm.startswith(key_norm) and len(key_norm) > best_len:
                        best_match = key
                        best_len = len(key_norm)
                if best_match:
                    files[best_match] = f
    return files


def map_tracebench_to_taxonomy(tb_labels, mapping):
    """Map TraceBench labels to our 8-dimension binary vector."""
    result = {dim: 0 for dim in OUR_DIMENSIONS}
    has_any_issue = False
    for label_code in tb_labels:
        if label_code == "NOL":
            continue
        info = mapping.get(label_code)
        if info and info.get("our_dimension"):
            result[info["our_dimension"]] = 1
            has_any_issue = True
    if not has_any_issue:
        result["healthy"] = 1
    return result


def preprocess_features(raw_features_list, config, scalers):
    """Apply our full preprocessing pipeline to raw features."""
    df = pd.DataFrame(raw_features_list)
    logger.info("Raw features shape: %s", df.shape)
    df = stage3_engineer(df)
    logger.info("After engineering: %s", df.shape)
    for col in DROP_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])
    df_norm, _ = stage5_normalize(df, config, fit=False, scalers=scalers)
    logger.info("After normalization: %s", df_norm.shape)
    return df_norm


def align_features(df, expected_cols):
    """Ensure df has exactly the expected columns in the right order."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected_cols].copy()
    return df


def find_optimal_threshold(y_true, y_proba, thresholds=None):
    """Find threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def loo_cv_threshold(y_true, y_proba):
    """Leave-one-out cross-validated threshold optimization.

    For each trace i, find optimal threshold on all other traces,
    then apply it to trace i. Returns the LOO predictions and the
    per-fold optimal thresholds.
    """
    n = len(y_true)
    loo_preds = np.zeros(n, dtype=int)
    loo_thresholds = np.zeros(n)
    thresholds = np.arange(0.01, 1.0, 0.01)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        y_train = y_true[mask]
        p_train = y_proba[mask]

        # Find optimal threshold on training fold
        opt_t, _ = find_optimal_threshold(y_train, p_train, thresholds)
        loo_thresholds[i] = opt_t
        loo_preds[i] = int(y_proba[i] >= opt_t)

    return loo_preds, loo_thresholds


def main():
    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Load pre-fitted scalers
    logger.info("Loading scalers from %s", SCALERS_PATH)
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)

    # Load model
    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    dim_names = list(models.keys())
    logger.info("Model dimensions: %s", dim_names)

    # Load label mapping
    with open(LABEL_MAPPING_PATH) as f:
        label_mapping_full = json.load(f)
    tb_to_taxonomy = label_mapping_full["tracebench_to_our_taxonomy"]

    # Load expected feature columns
    prod_features = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "production" / "features_normalized.parquet",
        columns=None,
    )
    expected_cols = [c for c in prod_features.columns if not c.startswith("_")]
    del prod_features
    logger.info("Expected %d features", len(expected_cols))

    # Load TraceBench labels & find darshan files
    all_labels = load_tracebench_labels()
    logger.info("Loaded labels for %d traces", len(all_labels))
    darshan_files = find_darshan_files(all_labels)
    logger.info("Found %d darshan files", len(darshan_files))

    # Parse each trace
    raw_features_list = []
    trace_keys = []
    for (subset, trace_name) in sorted(all_labels.keys()):
        darshan_path = darshan_files.get((subset, trace_name))
        if darshan_path is None:
            continue
        logger.info("Parsing %s/%s", subset, trace_name)
        features = parse_darshan_subprocess(str(darshan_path), timeout=120)
        if features is None:
            continue
        raw_features_list.append(features)
        trace_keys.append((subset, trace_name))

    logger.info("Successfully parsed %d / %d traces", len(raw_features_list), len(all_labels))
    if not raw_features_list:
        logger.error("No traces parsed. Exiting.")
        return

    # Preprocess
    df_processed = preprocess_features(raw_features_list, config, scalers)
    df_aligned = align_features(df_processed, expected_cols)
    logger.info("Aligned features shape: %s", df_aligned.shape)

    # Build ground truth
    y_true = np.zeros((len(trace_keys), len(dim_names)), dtype=int)
    for i, (subset, trace_name) in enumerate(trace_keys):
        tb_labels = all_labels[(subset, trace_name)]
        mapped = map_tracebench_to_taxonomy(tb_labels, tb_to_taxonomy)
        for j, dim in enumerate(dim_names):
            y_true[i, j] = mapped.get(dim, 0)

    # Get probabilities from each per-dimension model
    X = df_aligned.values.astype(np.float32)
    n_traces = X.shape[0]
    n_dims = len(dim_names)
    y_proba = np.zeros((n_traces, n_dims))

    for j, dim in enumerate(dim_names):
        model = models[dim]
        # XGBoost predict_proba returns [[P(0), P(1)], ...]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                y_proba[:, j] = proba[:, 1]
            else:
                y_proba[:, j] = proba.ravel()
        else:
            # Fallback to decision function or raw predict
            y_proba[:, j] = model.predict(X).astype(float)

    # ---------------------------------------------------------------------------
    # Per-dimension probability diagnostic
    # ---------------------------------------------------------------------------
    results = {"per_dimension": {}, "overall": {}}
    thresholds_grid = np.arange(0.01, 1.0, 0.01)

    print("\n" + "=" * 80)
    print("TRACEBENCH PROBABILITY DIAGNOSTIC")
    print("=" * 80)
    print(f"Traces: {n_traces} parsed")
    print(f"Dimensions: {n_dims} ({len(TRACEBENCH_COVERED_DIMS)} covered by TraceBench)")

    print(f"\n{'Dimension':<25s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'OptThr':>7s} "
          f"{'OptF1':>6s} {'LOO-F1':>7s} {'P(+)|TP':>8s} {'P(+)|TN':>8s} "
          f"{'Supp':>5s} {'Signal':>10s}")
    print("-" * 105)

    # Collect LOO predictions for overall metric
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = np.zeros_like(y_true)
    y_pred_loo = np.zeros_like(y_true)
    optimal_thresholds = {}

    for j, dim in enumerate(dim_names):
        yt = y_true[:, j]
        yp = y_proba[:, j]
        n_pos = int(yt.sum())
        n_neg = int((1 - yt).sum())

        dim_result = {
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

        # Probability stats
        if n_pos > 0:
            mean_proba_tp = float(yp[yt == 1].mean())
            dim_result["mean_proba_true_positive"] = round(mean_proba_tp, 4)
        else:
            mean_proba_tp = float("nan")

        if n_neg > 0:
            mean_proba_tn = float(yp[yt == 0].mean())
            dim_result["mean_proba_true_negative"] = round(mean_proba_tn, 4)
        else:
            mean_proba_tn = float("nan")

        # AUC-ROC
        if n_pos > 0 and n_neg > 0:
            auc_roc = roc_auc_score(yt, yp)
            dim_result["auc_roc"] = round(float(auc_roc), 4)
        else:
            auc_roc = float("nan")
            dim_result["auc_roc"] = None
            dim_result["auc_roc_note"] = "undefined (all same class)"

        # AUC-PR (average precision)
        if n_pos > 0:
            auc_pr = average_precision_score(yt, yp)
            dim_result["auc_pr"] = round(float(auc_pr), 4)
        else:
            auc_pr = float("nan")
            dim_result["auc_pr"] = None

        # Optimal threshold (on full data - oracle)
        if n_pos > 0 and n_neg > 0:
            opt_thresh, opt_f1 = find_optimal_threshold(yt, yp, thresholds_grid)
            dim_result["optimal_threshold"] = round(float(opt_thresh), 2)
            dim_result["optimal_f1"] = round(float(opt_f1), 4)
            optimal_thresholds[dim] = opt_thresh
            y_pred_optimal[:, j] = (yp >= opt_thresh).astype(int)
        elif n_pos > 0:
            # All positive: threshold = 0.0
            opt_thresh, opt_f1 = 0.01, 1.0
            optimal_thresholds[dim] = opt_thresh
            y_pred_optimal[:, j] = (yp >= opt_thresh).astype(int)
            dim_result["optimal_threshold"] = 0.01
            dim_result["optimal_f1"] = 1.0
        else:
            opt_thresh, opt_f1 = 0.99, float("nan")
            optimal_thresholds[dim] = 0.99
            y_pred_optimal[:, j] = 0
            dim_result["optimal_threshold"] = 0.99
            dim_result["optimal_f1"] = None

        # LOO-CV threshold
        if n_pos > 0 and n_neg > 0 and n_pos >= 2:
            loo_preds, loo_thresholds = loo_cv_threshold(yt, yp)
            y_pred_loo[:, j] = loo_preds
            tp_loo = ((loo_preds == 1) & (yt == 1)).sum()
            fp_loo = ((loo_preds == 1) & (yt == 0)).sum()
            fn_loo = ((loo_preds == 0) & (yt == 1)).sum()
            prec_loo = tp_loo / max(tp_loo + fp_loo, 1)
            rec_loo = tp_loo / max(tp_loo + fn_loo, 1)
            loo_f1 = 2 * prec_loo * rec_loo / max(prec_loo + rec_loo, 1e-9)
            dim_result["loo_cv_f1"] = round(float(loo_f1), 4)
            dim_result["loo_cv_mean_threshold"] = round(float(loo_thresholds.mean()), 3)
        else:
            loo_f1 = float("nan")
            y_pred_loo[:, j] = y_pred_default[:, j]
            dim_result["loo_cv_f1"] = None
            dim_result["loo_cv_note"] = "insufficient positive/negative examples for LOO"

        # Signal assessment
        if not np.isnan(auc_roc):
            if auc_roc >= 0.7:
                signal = "STRONG"
            elif auc_roc >= 0.5:
                signal = "PARTIAL"
            else:
                signal = "INVERSE"
        else:
            signal = "N/A"
        dim_result["signal_assessment"] = signal

        results["per_dimension"][dim] = dim_result

        # Print row
        covered = "*" if dim not in TRACEBENCH_COVERED_DIMS else " "
        auc_roc_str = f"{auc_roc:.4f}" if not np.isnan(auc_roc) else "  N/A  "
        auc_pr_str = f"{auc_pr:.4f}" if not np.isnan(auc_pr) else "  N/A  "
        opt_f1_str = f"{opt_f1:.3f}" if not np.isnan(opt_f1) else " N/A "
        loo_f1_str = f"{loo_f1:.4f}" if not np.isnan(loo_f1) else "  N/A  "
        tp_str = f"{mean_proba_tp:.4f}" if not np.isnan(mean_proba_tp) else "  N/A  "
        tn_str = f"{mean_proba_tn:.4f}" if not np.isnan(mean_proba_tn) else "  N/A  "

        print(f"{covered}{dim:<24s} {auc_roc_str:>8s} {auc_pr_str:>8s} {opt_thresh:>7.2f} "
              f"{opt_f1_str:>6s} {loo_f1_str:>7s} {tp_str:>8s} {tn_str:>8s} "
              f"{n_pos:>5d} {signal:>10s}")

    print("-" * 105)
    print("* = not covered by TraceBench labels\n")

    # ---------------------------------------------------------------------------
    # Overall metrics at different threshold strategies
    # ---------------------------------------------------------------------------
    # Only on covered dimensions
    cov_idx = [i for i, d in enumerate(dim_names) if d in TRACEBENCH_COVERED_DIMS]

    y_true_cov = y_true[:, cov_idx]
    y_pred_default_cov = y_pred_default[:, cov_idx]
    y_pred_optimal_cov = y_pred_optimal[:, cov_idx]
    y_pred_loo_cov = y_pred_loo[:, cov_idx]

    micro_f1_default = f1_score(y_true_cov, y_pred_default_cov, average="micro", zero_division=0)
    macro_f1_default = f1_score(y_true_cov, y_pred_default_cov, average="macro", zero_division=0)

    micro_f1_optimal = f1_score(y_true_cov, y_pred_optimal_cov, average="micro", zero_division=0)
    macro_f1_optimal = f1_score(y_true_cov, y_pred_optimal_cov, average="macro", zero_division=0)

    micro_f1_loo = f1_score(y_true_cov, y_pred_loo_cov, average="micro", zero_division=0)
    macro_f1_loo = f1_score(y_true_cov, y_pred_loo_cov, average="macro", zero_division=0)

    results["overall"] = {
        "default_threshold_0.5": {
            "micro_f1": round(float(micro_f1_default), 4),
            "macro_f1": round(float(macro_f1_default), 4),
        },
        "oracle_optimal_threshold": {
            "micro_f1": round(float(micro_f1_optimal), 4),
            "macro_f1": round(float(macro_f1_optimal), 4),
            "thresholds": {d: round(float(optimal_thresholds[d]), 2) for d in dim_names},
        },
        "loo_cv_threshold": {
            "micro_f1": round(float(micro_f1_loo), 4),
            "macro_f1": round(float(macro_f1_loo), 4),
        },
        "n_traces": n_traces,
    }

    print("=" * 80)
    print("OVERALL METRICS (6 covered dimensions)")
    print("=" * 80)
    print(f"  {'Strategy':<30s} {'Micro-F1':>10s} {'Macro-F1':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Default (threshold=0.5)':<30s} {micro_f1_default:>10.4f} {macro_f1_default:>10.4f}")
    print(f"  {'Oracle optimal threshold':<30s} {micro_f1_optimal:>10.4f} {macro_f1_optimal:>10.4f}")
    print(f"  {'LOO-CV threshold':<30s} {micro_f1_loo:>10.4f} {macro_f1_loo:>10.4f}")

    # Improvement ratio
    if micro_f1_default > 0:
        improvement = micro_f1_loo / micro_f1_default
        print(f"\n  LOO-CV improvement over default: {improvement:.2f}x")
    else:
        print(f"\n  LOO-CV Micro-F1: {micro_f1_loo:.4f} (default was 0)")

    # ---------------------------------------------------------------------------
    # Probability distribution summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    for dim in TRACEBENCH_COVERED_DIMS:
        j = dim_names.index(dim)
        yt = y_true[:, j]
        yp = y_proba[:, j]
        n_pos = int(yt.sum())
        n_neg = int((1 - yt).sum())
        print(f"\n  {dim} (support={n_pos}):")
        if n_pos > 0:
            probs_tp = yp[yt == 1]
            print(f"    True positives  P(+): mean={probs_tp.mean():.4f}, "
                  f"min={probs_tp.min():.4f}, max={probs_tp.max():.4f}, "
                  f"median={np.median(probs_tp):.4f}")
        if n_neg > 0:
            probs_tn = yp[yt == 0]
            print(f"    True negatives  P(+): mean={probs_tn.mean():.4f}, "
                  f"min={probs_tn.min():.4f}, max={probs_tn.max():.4f}, "
                  f"median={np.median(probs_tn):.4f}")
        if n_pos > 0 and n_neg > 0:
            sep = yp[yt == 1].mean() - yp[yt == 0].mean()
            print(f"    Separation (mean TP - mean TN): {sep:+.4f}")

    # ---------------------------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------------------------
    covered_aucs = []
    for dim in TRACEBENCH_COVERED_DIMS:
        auc_val = results["per_dimension"][dim].get("auc_roc")
        if auc_val is not None:
            covered_aucs.append(auc_val)

    if covered_aucs:
        mean_auc = np.mean(covered_aucs)
    else:
        mean_auc = float("nan")

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    print(f"  Mean AUC-ROC (covered dims with both classes): {mean_auc:.4f}")
    print(f"  Number of dims with computable AUC: {len(covered_aucs)}/6")

    if not np.isnan(mean_auc):
        if mean_auc >= 0.7:
            verdict = "STRONG signal. Threshold recalibration will work well."
        elif mean_auc >= 0.5:
            verdict = "PARTIAL signal. Some improvement possible with threshold tuning."
        else:
            verdict = "NO signal. Model features do not transfer to TraceBench."
        print(f"  Assessment: {verdict}")

    results["verdict"] = {
        "mean_auc_roc_covered": round(float(mean_auc), 4) if not np.isnan(mean_auc) else None,
        "n_dims_with_auc": len(covered_aucs),
    }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "probability_diagnostic.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
