#!/usr/bin/env python
"""
TraceBench Evaluation Script
=============================
Evaluates our IOPrescriber ML classifier on the TraceBench dataset (35 Darshan
traces from IOAgent's evaluation suite) to measure generalization to external data.

TraceBench has 3 subsets:
  - single_issue_bench: 9 synthetic traces with injected issues
  - IO500: 17 IOR/mdtest traces from IO500 benchmark suite
  - real_app_bench: 9 real HPC application traces

We parse each trace, extract features using our pipeline, apply preprocessing
(engineering + normalization), and run our trained XGBoost biquality model.

Labels are mapped from TraceBench's fine-grained categories to our 8-dimension
taxonomy using label_mapping.json.
"""

import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup paths
PROJECT_ROOT = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_extraction import extract_raw_features, FEATURE_GROUPS
from src.data.preprocessing import stage3_engineer, stage5_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — BOOST EXPERIMENT: use new model, save to isolated dir
# ---------------------------------------------------------------------------
TRACEBENCH_ROOT = PROJECT_ROOT / "data" / "external" / "tracebench" / "TraceBench"
LABEL_MAPPING_PATH = PROJECT_ROOT / "data" / "external" / "tracebench" / "label_mapping.json"
MODEL_PATH = PROJECT_ROOT / "results" / "boost_experiment" / "new_models" / "xgboost_biquality_w100_seed42.pkl"
SCALERS_PATH = PROJECT_ROOT / "data" / "processed" / "production" / "scalers.pkl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "boost_experiment" / "full_evaluation" / "tracebench"

# Our 8-dimension taxonomy (model output labels)
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

# Dimensions covered by TraceBench (6 of 8)
TRACEBENCH_COVERED_DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
]

# Features expected by the model (157 features, no _ prefixed metadata)
# This list is derived from the production normalized features parquet
EXPECTED_FEATURES = None  # Will be loaded from production data

# Features to drop (from preprocessing.yaml feature_exclusion)
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
    # Auto-dropped constants
    "POSIX_MEM_ALIGNMENT",
    "POSIX_MMAPS",
    "POSIX_FDSYNCS",
    "has_posix",
    "has_hdf5",
    "has_pnetcdf",
    "has_heatmap",
    # Variance counters (often zero/constant)
    "POSIX_F_VARIANCE_RANK_TIME",
    "POSIX_F_VARIANCE_RANK_BYTES",
    "MPIIO_F_VARIANCE_RANK_TIME",
    "MPIIO_F_VARIANCE_RANK_BYTES",
    "STDIO_F_VARIANCE_RANK_TIME",
    "STDIO_F_VARIANCE_RANK_BYTES",
    # MPI-IO counters that were dropped
    "MPIIO_SPLIT_READS",
    "MPIIO_SPLIT_WRITES",
    "MPIIO_NB_READS",
    # Derived features that may not appear
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
# Convert to JSON-serializable
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
    labels = {}  # trace_name -> list of TraceBench label codes

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
    """Find all darshan files across TraceBench subsets, matched to trace names.

    Uses substring matching: a darshan filename matches a label key if the
    label key (with spaces replaced by underscores) is a prefix of the filename
    (also with spaces replaced by underscores).
    """
    files = {}  # (subset, trace_name) -> darshan_path

    for subset in ["single_issue_bench", "IO500", "real_app_bench"]:
        darshan_dir = TRACEBENCH_ROOT / "Datasets" / subset / "darshan_files" / "darshan"
        if not darshan_dir.exists():
            logger.warning("No darshan dir for %s", subset)
            continue

        # Get all label keys for this subset
        subset_keys = [k for k in all_labels if k[0] == subset]

        for f in sorted(darshan_dir.glob("*.darshan")):
            fname = f.stem
            if fname.endswith(".ini"):
                fname = fname[:-4]

            # Normalize: replace spaces with underscores for comparison
            fname_norm = fname.replace(" ", "_")

            # Try exact match first
            for key in subset_keys:
                key_norm = key[1].replace(" ", "_")
                if fname_norm == key_norm:
                    files[key] = f
                    break
            else:
                # Try prefix match (label key is prefix of filename)
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
    """Map TraceBench labels to our 8-dimension binary vector.

    Parameters
    ----------
    tb_labels : list of str
        TraceBench label codes (e.g., ["SML-R", "MSL-W", "HMD"])
    mapping : dict
        The tracebench_to_our_taxonomy mapping from label_mapping.json

    Returns
    -------
    dict
        {dimension: 0 or 1} for each of our 8 dimensions
    """
    result = {dim: 0 for dim in OUR_DIMENSIONS}

    has_any_issue = False
    for label_code in tb_labels:
        if label_code == "NOL":
            continue
        info = mapping.get(label_code)
        if info and info.get("our_dimension"):
            result[info["our_dimension"]] = 1
            has_any_issue = True

    # healthy = 1 if no issues detected
    if not has_any_issue:
        result["healthy"] = 1

    return result


def preprocess_features(raw_features_list, config, scalers):
    """Apply our full preprocessing pipeline to raw features.

    Steps:
    1. Create DataFrame from raw feature dicts
    2. Apply stage3 engineering (derived features)
    3. Drop excluded features
    4. Apply stage5 normalization with pre-fitted scalers
    5. Align columns with model expectation
    """
    df = pd.DataFrame(raw_features_list)
    logger.info("Raw features shape: %s", df.shape)

    # Stage 3: Engineer derived features
    df = stage3_engineer(df)
    logger.info("After engineering: %s", df.shape)

    # Drop excluded features
    for col in DROP_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Stage 5: Normalize using pre-fitted scalers
    df_norm, _ = stage5_normalize(df, config, fit=False, scalers=scalers)
    logger.info("After normalization: %s", df_norm.shape)

    return df_norm


def align_features(df, expected_cols):
    """Ensure df has exactly the expected columns in the right order."""
    # Add missing columns as 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.debug("Added missing feature: %s", col)

    # Keep only expected columns in order
    df = df[expected_cols].copy()
    return df


def compute_metrics(y_true, y_pred, dim_names):
    """Compute per-dimension and overall metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        precision_recall_fscore_support, classification_report,
    )

    results = {}

    # Per-dimension metrics
    per_dim = {}
    for i, dim in enumerate(dim_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        n_pos = yt.sum()
        n_pred = yp.sum()
        tp = (yt * yp).sum()

        prec = tp / max(n_pred, 1)
        rec = tp / max(n_pos, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        per_dim[dim] = {
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1": round(float(f1), 3),
            "support": int(n_pos),
            "predicted_positive": int(n_pred),
            "true_positive": int(tp),
        }

    results["per_dimension"] = per_dim

    # Overall metrics (micro and macro)
    # Only evaluate on TraceBench-covered dimensions (6 of 8)
    covered_idx = [i for i, d in enumerate(dim_names) if d in TRACEBENCH_COVERED_DIMS]
    y_true_cov = y_true[:, covered_idx]
    y_pred_cov = y_pred[:, covered_idx]

    micro_f1 = f1_score(y_true_cov, y_pred_cov, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_cov, y_pred_cov, average="macro", zero_division=0)
    samples_f1 = f1_score(y_true_cov, y_pred_cov, average="samples", zero_division=0)

    results["overall"] = {
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "samples_f1": round(float(samples_f1), 4),
        "n_traces": int(y_true.shape[0]),
        "n_dimensions_evaluated": len(covered_idx),
    }

    # Exact-match accuracy (IOAgent style: all dimensions match)
    exact_match = (y_true_cov == y_pred_cov).all(axis=1).mean()
    results["overall"]["exact_match_accuracy"] = round(float(exact_match), 4)

    # Subset accuracy (at least one correct positive prediction per trace)
    subset_correct = 0
    for i in range(y_true_cov.shape[0]):
        if y_true_cov[i].sum() == 0:
            # No issues in ground truth; correct if we also predict no issues
            if y_pred_cov[i].sum() == 0:
                subset_correct += 1
        else:
            # Has issues; correct if we get at least one right
            if (y_true_cov[i] * y_pred_cov[i]).sum() > 0:
                subset_correct += 1
    results["overall"]["partial_match_accuracy"] = round(
        float(subset_correct / max(y_true_cov.shape[0], 1)), 4
    )

    return results


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
    logger.info("Model dimensions: %s", list(models.keys()))

    # Load label mapping
    with open(LABEL_MAPPING_PATH) as f:
        label_mapping_full = json.load(f)
    tb_to_taxonomy = label_mapping_full["tracebench_to_our_taxonomy"]

    # Load expected feature columns from production data
    prod_features = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "production" / "features_normalized.parquet",
        columns=None,
    )
    expected_cols = [c for c in prod_features.columns if not c.startswith("_")]
    del prod_features
    logger.info("Expected %d features", len(expected_cols))

    # Load TraceBench labels
    all_labels = load_tracebench_labels()
    logger.info("Loaded labels for %d traces", len(all_labels))

    # Find darshan files
    darshan_files = find_darshan_files(all_labels)
    logger.info("Found %d darshan files", len(darshan_files))

    # Parse each trace
    raw_features_list = []
    trace_keys = []
    parse_errors = []

    for (subset, trace_name) in sorted(all_labels.keys()):
        # Find the darshan file
        darshan_path = darshan_files.get((subset, trace_name))
        if darshan_path is None:
            logger.warning("No darshan file for %s/%s", subset, trace_name)
            parse_errors.append({
                "subset": subset,
                "trace": trace_name,
                "error": "darshan file not found",
            })
            continue

        logger.info("Parsing %s/%s: %s", subset, trace_name, darshan_path.name)
        features = parse_darshan_subprocess(str(darshan_path), timeout=120)

        if features is None:
            logger.warning("Parse failed for %s/%s", subset, trace_name)
            parse_errors.append({
                "subset": subset,
                "trace": trace_name,
                "error": "parse failed",
            })
            continue

        raw_features_list.append(features)
        trace_keys.append((subset, trace_name))

    logger.info("Successfully parsed %d / %d traces", len(raw_features_list), len(all_labels))

    if not raw_features_list:
        logger.error("No traces parsed successfully. Exiting.")
        return

    # Preprocess features
    df_processed = preprocess_features(raw_features_list, config, scalers)

    # Align to expected columns
    df_aligned = align_features(df_processed, expected_cols)
    logger.info("Aligned features shape: %s", df_aligned.shape)

    # Build ground truth labels
    dim_names = list(models.keys())  # Order from model dict
    y_true = np.zeros((len(trace_keys), len(dim_names)), dtype=int)
    for i, (subset, trace_name) in enumerate(trace_keys):
        tb_labels = all_labels[(subset, trace_name)]
        mapped = map_tracebench_to_taxonomy(tb_labels, tb_to_taxonomy)
        for j, dim in enumerate(dim_names):
            y_true[i, j] = mapped.get(dim, 0)

    # Run predictions
    X = df_aligned.values.astype(np.float32)
    y_pred = np.zeros_like(y_true)

    for j, dim in enumerate(dim_names):
        model = models[dim]
        preds = model.predict(X)
        y_pred[:, j] = preds.astype(int)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, dim_names)

    # Per-trace results
    trace_results = []
    for i, (subset, trace_name) in enumerate(trace_keys):
        tb_labels = all_labels[(subset, trace_name)]
        pred_dims = [dim_names[j] for j in range(len(dim_names)) if y_pred[i, j] == 1]
        true_dims = [dim_names[j] for j in range(len(dim_names)) if y_true[i, j] == 1]
        correct = all(y_true[i, j] == y_pred[i, j] for j in range(len(dim_names))
                       if dim_names[j] in TRACEBENCH_COVERED_DIMS)
        trace_results.append({
            "subset": subset,
            "trace": trace_name,
            "tracebench_labels": tb_labels,
            "true_dimensions": true_dims,
            "predicted_dimensions": pred_dims,
            "exact_match": correct,
        })

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_results = {
        "model": "xgboost_biquality_w100",
        "dataset": "TraceBench v0.1.0",
        "n_traces_total": len(all_labels),
        "n_traces_parsed": len(trace_keys),
        "n_parse_errors": len(parse_errors),
        "metrics": metrics,
        "per_trace": trace_results,
        "parse_errors": parse_errors,
    }

    output_path = OUTPUT_DIR / "tracebench_results.json"
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("TraceBench Evaluation Results")
    print("=" * 70)
    print(f"Model: xgboost_biquality_w100 (Phase 2, biquality learning)")
    print(f"Traces: {len(trace_keys)} parsed / {len(all_labels)} total")
    print(f"Parse errors: {len(parse_errors)}")

    print(f"\n--- Overall Metrics (6 covered dimensions) ---")
    ov = metrics["overall"]
    print(f"  Micro-F1:              {ov['micro_f1']:.4f}")
    print(f"  Macro-F1:              {ov['macro_f1']:.4f}")
    print(f"  Samples-F1:            {ov['samples_f1']:.4f}")
    print(f"  Exact-match accuracy:  {ov['exact_match_accuracy']:.4f}")
    print(f"  Partial-match accuracy: {ov['partial_match_accuracy']:.4f}")

    print(f"\n--- Per-Dimension Metrics ---")
    print(f"  {'Dimension':<25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s} {'Pred+':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for dim in dim_names:
        d = metrics["per_dimension"][dim]
        marker = "" if dim in TRACEBENCH_COVERED_DIMS else " *"
        print(f"  {dim+marker:<25s} {d['precision']:>6.3f} {d['recall']:>6.3f} "
              f"{d['f1']:>6.3f} {d['support']:>8d} {d['predicted_positive']:>6d}")
    print(f"  (* = not covered by TraceBench labels)")

    print(f"\n--- Per-Trace Results ---")
    for tr in trace_results:
        status = "CORRECT" if tr["exact_match"] else "WRONG"
        print(f"  [{status:>7s}] {tr['subset']}/{tr['trace'][:50]}")
        print(f"           True:  {tr['true_dimensions']}")
        print(f"           Pred:  {tr['predicted_dimensions']}")

    # IOAgent comparison
    print(f"\n--- Comparison with IOAgent (IPDPS'25) ---")
    print(f"  IOAgent reported accuracy: 0.641 (exact-match on TraceBench)")
    print(f"  Our exact-match accuracy:  {ov['exact_match_accuracy']:.4f}")
    print(f"  Our Micro-F1:              {ov['micro_f1']:.4f}")
    print(f"  Our Macro-F1:              {ov['macro_f1']:.4f}")
    print(f"  Note: We evaluate on 6/8 dimensions (TraceBench does not cover")
    print(f"        throughput_utilization and temporal_pattern)")


if __name__ == "__main__":
    main()
