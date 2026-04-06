#!/usr/bin/env python3
"""
Boost Experiment: Process new benchmark logs and compare old vs new GT results.

This script is FULLY ISOLATED — it does NOT modify any existing data files,
models, KB, or results. Everything is saved to results/boost_experiment/.

Steps:
  1. Map new Darshan logs to their config labels via SLURM step timestamps
  2. Parse and extract features from new logs
  3. Merge with existing 623 GT samples → new combined GT dataset
  4. Re-split using iterative stratification (same methodology, same seed)
  5. Retrain XGBoost biquality models (5 seeds) on new splits
  6. Evaluate on new test set
  7. Run ablation studies
  8. Compare old vs new results side-by-side

Usage:
    python scripts/run_boost_experiment.py
"""

import argparse
import copy
import json
import logging
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_DIR / "results" / "boost_experiment"

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

# ============================================================================
# Step 1: Map Darshan logs to labels via SLURM step timestamps
# ============================================================================

# Job 17310653 (boost_weak_dims): 66 steps
# Steps 0-23:  access_pattern (24 runs) — IOR with -z flag (random access)
# Steps 24-41: file_strategy  (18 runs) — IOR with -F flag (file-per-process)
# Steps 42-59: throughput_utilization (18 runs) — IOR with -Y flag (fsync)
# Steps 60-65: healthy (6 runs) — IOR with MPIIO collective

BOOST_JOB_LABELS = {}
for i in range(0, 24):
    BOOST_JOB_LABELS[f"17310653.{i}"] = {"access_pattern": 1}
for i in range(24, 42):
    BOOST_JOB_LABELS[f"17310653.{i}"] = {"file_strategy": 1}
for i in range(42, 60):
    BOOST_JOB_LABELS[f"17310653.{i}"] = {"throughput_utilization": 1}
for i in range(60, 66):
    BOOST_JOB_LABELS[f"17310653.{i}"] = {"healthy": 1}

# Job 17309295 (ior_random_boost — timed out, 33/66 completed):
# All configs are random access (-z flag) — access_pattern=1
# Some with small sizes (512, 1024, 2048) also get access_granularity=1
# The script ran: random_fpp (5 sizes × 2 ranks × 3 reps = 30) then random_shared (started)
# Steps 0-29: random_fpp (various sizes, all -z -F)
# Steps 30-32: random_shared (started, t=4096 n=4 rep 1,2,3 — only 3 completed before timeout)
# All are access_pattern=1, some with access_granularity=1 for small sizes
# We'll label them all as access_pattern=1 (conservative) and verify via Darshan features

RANDOM_BOOST_LABELS = {}
for i in range(0, 33):
    RANDOM_BOOST_LABELS[f"17309295.{i}"] = {"access_pattern": 1}


def get_slurm_step_times(job_id):
    """Get start/end times for each SLURM job step."""
    result = subprocess.run(
        ["sacct", "-j", str(job_id),
         "--format=JobID%25,Start,End", "--noheader", "--parsable2"],
        capture_output=True, text=True, timeout=30
    )
    steps = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 3:
            step_id = parts[0].strip()
            start = parts[1].strip()
            end = parts[2].strip()
            # Only keep numbered steps (not batch/extern)
            if re.match(r"\d+\.\d+$", step_id):
                steps[step_id] = {
                    "start": datetime.strptime(start, "%Y-%m-%dT%H:%M:%S"),
                    "end": datetime.strptime(end, "%Y-%m-%dT%H:%M:%S"),
                }
    return steps


def map_darshan_to_step(darshan_path, step_times):
    """Map a Darshan log to its SLURM step by matching job ID embedded in filename.

    Darshan filenames contain the SLURM step ID:
      mbanisha_ior_id{JOBID}-{STEPID}_{date}...
    The STEPID maps directly to the srun step number.
    """
    basename = os.path.basename(darshan_path)
    # Extract job ID and step/rank ID from filename
    match = re.search(r"_id(\d+)-(\d+)_", basename)
    if not match:
        return None
    job_id = match.group(1)
    step_internal_id = match.group(2)

    # The internal step ID in Darshan doesn't directly match sacct step number.
    # Instead, we use the file modification time to find the matching step.
    try:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(darshan_path))
    except OSError:
        return None

    # Find the step whose time range contains this file's mtime
    best_step = None
    best_delta = None
    for step_id, times in step_times.items():
        if not step_id.startswith(job_id + "."):
            continue
        # Check if file mtime is within step's time window (with 60s buffer)
        from datetime import timedelta
        start = times["start"] - timedelta(seconds=60)
        end = times["end"] + timedelta(seconds=60)
        if start <= file_mtime <= end:
            delta = abs((file_mtime - times["end"]).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_step = step_id
    return best_step


def map_darshan_logs_to_labels(job_id, label_map, log_dir):
    """Map all Darshan logs from a job to their construction-based labels."""
    step_times = get_slurm_step_times(job_id)
    logger.info("Got %d SLURM steps for job %s", len(step_times), job_id)

    log_dir = Path(log_dir)
    darshan_files = sorted(log_dir.glob(f"*{job_id}*.darshan"))
    logger.info("Found %d Darshan logs for job %s", len(darshan_files), job_id)

    mappings = []
    unmapped = 0
    for fpath in darshan_files:
        step = map_darshan_to_step(str(fpath), step_times)
        if step and step in label_map:
            labels = {dim: 0 for dim in DIMENSIONS}
            labels.update(label_map[step])
            # Set healthy flag
            if sum(v for k, v in labels.items() if k != "healthy") == 0:
                labels["healthy"] = 1
            else:
                labels["healthy"] = 0
            mappings.append({
                "path": str(fpath),
                "filename": fpath.name,
                "step_id": step,
                "labels": labels,
                "job_id": job_id,
            })
        else:
            unmapped += 1
            logger.warning("Could not map %s (step=%s)", fpath.name, step)

    logger.info("Mapped: %d, Unmapped: %d", len(mappings), unmapped)
    return mappings


# ============================================================================
# Step 2: Parse and extract features
# ============================================================================

def extract_features_from_logs(mappings):
    """Parse Darshan logs and extract features."""
    from src.data.parse_darshan import parse_darshan_log
    from src.data.feature_extraction import extract_raw_features
    from src.data.preprocessing import stage3_engineer

    feature_rows = []
    label_rows = []
    n_ok = 0
    n_fail = 0

    for m in mappings:
        try:
            parsed = parse_darshan_log(m["path"])
            if parsed is None:
                n_fail += 1
                continue
            features = extract_raw_features(parsed)
            features["_source_path"] = m["path"]
            features["_benchmark"] = "ior"
            features["_scenario"] = m["step_id"]
            features["_ground_truth_job_id"] = m["job_id"]
            feature_rows.append(features)

            label_row = {
                "job_id": m["job_id"],
                "benchmark": "ior",
                "scenario": m["step_id"],
                "n_ranks": features.get("nprocs", 1),
                "n_darshan_files": 1,
                "label_source": "construction",
            }
            label_row.update(m["labels"])
            label_rows.append(label_row)
            n_ok += 1
        except Exception as e:
            logger.warning("Failed to parse %s: %s", m["filename"], e)
            n_fail += 1

    logger.info("Extracted: %d OK, %d failed", n_ok, n_fail)

    if not feature_rows:
        return None, None

    features_df = pd.DataFrame(feature_rows)
    labels_df = pd.DataFrame(label_rows)

    # Apply feature engineering (same 39 derived features as production)
    extra_cols = ["_source_path", "_benchmark", "_scenario", "_ground_truth_job_id"]
    extra_data = {col: features_df[col] for col in extra_cols if col in features_df.columns}
    try:
        features_df = stage3_engineer(features_df)
        logger.info("Applied feature engineering: %d columns", len(features_df.columns))
    except Exception as e:
        logger.warning("Feature engineering failed (%s), using raw features", e)
    for col, data in extra_data.items():
        if col not in features_df.columns:
            features_df[col] = data.values

    # Ensure label column order
    label_meta = ["job_id", "benchmark", "scenario", "n_ranks", "n_darshan_files", "label_source"]
    labels_df = labels_df[label_meta + DIMENSIONS]

    return features_df, labels_df


# ============================================================================
# Step 3-4: Merge with existing GT and re-split
# ============================================================================

def merge_and_split(new_features, new_labels, output_dir, seed=42, test_ratio=0.7):
    """Merge new logs with existing 623 GT, re-split, save to isolated dir."""
    bench_dir = PROJECT_DIR / "data" / "processed" / "benchmark"

    # Load existing GT
    old_features = pd.read_parquet(bench_dir / "features.parquet")
    old_labels = pd.read_parquet(bench_dir / "labels.parquet")
    logger.info("Existing GT: %d samples", len(old_features))

    # Align columns (new features may have different columns)
    # Use old_features columns as reference, fill missing with 0
    for col in old_features.columns:
        if col not in new_features.columns:
            new_features[col] = 0
    # Keep only columns in old_features (same order)
    new_features = new_features[old_features.columns]

    # Similarly for labels
    for col in old_labels.columns:
        if col not in new_labels.columns:
            new_labels[col] = "" if old_labels[col].dtype == object else 0
    new_labels = new_labels[old_labels.columns]

    # Merge
    combined_features = pd.concat([old_features, new_features], ignore_index=True)
    combined_labels = pd.concat([old_labels, new_labels], ignore_index=True)
    logger.info("Combined GT: %d samples (old %d + new %d)",
                len(combined_features), len(old_features), len(new_features))

    # Re-split using iterative stratification (same methodology as prepare_phase2_data.py)
    y = combined_labels[DIMENSIONS].values.astype(int)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    for dev_idx, test_idx in msss.split(combined_features.values[:, :10], y):
        pass

    logger.info("New split: %d dev, %d test", len(dev_idx), len(test_idx))

    # Log label distributions
    y_dev = y[dev_idx]
    y_test = y[test_idx]
    logger.info("Label distribution in new test set:")
    for i, dim in enumerate(DIMENSIONS):
        logger.info("  %s: dev=%d, test=%d (was test=%d)",
                    dim, y_dev[:, i].sum(), y_test[:, i].sum(),
                    int(old_labels.iloc[:len(old_labels)][dim].sum()
                        if dim in old_labels.columns else 0))

    # Save to isolated directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_features.to_parquet(output_dir / "features.parquet", index=False)
    combined_labels.to_parquet(output_dir / "labels.parquet", index=False)

    with open(output_dir / "split_indices.pkl", "wb") as f:
        pickle.dump({"dev_idx": dev_idx, "test_idx": test_idx}, f)

    # Save test set separately
    test_features = combined_features.iloc[test_idx]
    test_labels = combined_labels.iloc[test_idx]
    test_features.to_parquet(output_dir / "test_features.parquet", index=False)
    test_labels.to_parquet(output_dir / "test_labels.parquet", index=False)

    return combined_features, combined_labels, dev_idx, test_idx


# ============================================================================
# Step 5: Retrain models
# ============================================================================

def train_models(gt_dir, output_model_dir, seeds=[42, 123, 456, 789, 1024],
                 clean_weight=100.0):
    """Train XGBoost biquality models on new combined GT data.

    Memory-optimized: loads data once, trains sequentially per seed,
    uses n_jobs=4 to limit parallel memory, no eval_set to save RAM.
    """
    import gc
    from xgboost import XGBClassifier

    config = yaml.safe_load(open(PROJECT_DIR / "configs" / "training.yaml"))

    # Load production data (unchanged — same 91K heuristic labels)
    paths = config["paths"]
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

    X_prod_train = prod_features.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_prod_train = prod_labels.iloc[train_idx][DIMENSIONS].values.astype(np.float32)

    # Free memory from full dataframes
    del prod_features, prod_labels
    gc.collect()

    # Load NEW benchmark data
    gt_dir = Path(gt_dir)
    bench_features = pd.read_parquet(gt_dir / "features.parquet")
    bench_labels = pd.read_parquet(gt_dir / "labels.parquet")
    with open(gt_dir / "split_indices.pkl", "rb") as f:
        bench_splits = pickle.load(f)
    dev_idx = bench_splits["dev_idx"]
    test_idx = bench_splits["test_idx"]

    # Align benchmark features to production feature columns
    X_bench_dev_cols = []
    for col in feature_cols:
        if col in bench_features.columns:
            X_bench_dev_cols.append(bench_features.iloc[dev_idx][col].values)
        else:
            X_bench_dev_cols.append(np.zeros(len(dev_idx)))
    X_bench_dev = np.column_stack(X_bench_dev_cols).astype(np.float32)
    y_bench_dev = bench_labels.iloc[dev_idx][DIMENSIONS].values.astype(np.float32)

    # Also prepare test set features for evaluation
    X_bench_test_cols = []
    for col in feature_cols:
        if col in bench_features.columns:
            X_bench_test_cols.append(bench_features.iloc[test_idx][col].values)
        else:
            X_bench_test_cols.append(np.zeros(len(test_idx)))
    X_bench_test = np.column_stack(X_bench_test_cols).astype(np.float32)
    y_bench_test = bench_labels.iloc[test_idx][DIMENSIONS].values.astype(np.float32)

    del bench_features, bench_labels
    gc.collect()

    # XGBoost params — memory optimized
    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params["n_jobs"] = 4  # Limit parallelism to reduce memory
    # Remove eval_metric to skip eval_set requirement
    xgb_params.pop("eval_metric", None)

    # Combine production + benchmark dev (once, reuse across seeds)
    n_prod_train = len(X_prod_train)
    n_bench_dev = len(X_bench_dev)
    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])
    weights = np.ones(len(X_combined), dtype=np.float32)
    weights[-n_bench_dev:] = clean_weight

    del X_prod_train, y_prod_train, X_bench_dev, y_bench_dev
    gc.collect()

    logger.info("Combined training set: %d samples (%d features)",
                len(X_combined), X_combined.shape[1])

    # Per-label scale_pos_weight
    spw = []
    for i in range(y_combined.shape[1]):
        n_pos = y_combined[:, i].sum()
        n_neg = len(y_combined) - n_pos
        spw.append(min(n_neg / max(n_pos, 1), 100.0))

    # Train with multiple seeds
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        logger.info("Training with seed=%d ...", seed)

        models = {}
        for i, dim in enumerate(DIMENSIONS):
            clf = XGBClassifier(
                **xgb_params, scale_pos_weight=spw[i],
                random_state=seed, verbosity=0,
            )
            clf.fit(
                X_combined, y_combined[:, i],
                sample_weight=weights,
                verbose=False,
            )
            models[dim] = clf

        # Save model
        model_path = output_model_dir / f"xgboost_biquality_w{int(clean_weight)}_seed{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

        # Evaluate on NEW test set
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_pred = np.zeros_like(y_bench_test)
        y_prob = np.zeros_like(y_bench_test)
        for i, dim in enumerate(DIMENSIONS):
            y_prob[:, i] = models[dim].predict_proba(X_bench_test)[:, 1]
            y_pred[:, i] = (y_prob[:, i] >= 0.5).astype(int)

        micro_f1 = f1_score(y_bench_test, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_bench_test, y_pred, average="macro", zero_division=0)

        per_dim = {}
        for i, dim in enumerate(DIMENSIONS):
            per_dim[dim] = {
                "f1": float(f1_score(y_bench_test[:, i], y_pred[:, i], zero_division=0)),
                "precision": float(precision_score(y_bench_test[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_bench_test[:, i], y_pred[:, i], zero_division=0)),
                "support": int(y_bench_test[:, i].sum()),
            }

        result = {
            "seed": seed,
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "per_dimension": per_dim,
            "n_train_prod": n_prod_train,
            "n_train_bench_dev": n_bench_dev,
            "n_test": len(X_bench_test),
        }
        all_results.append(result)
        logger.info("  Seed %d: Micro-F1=%.4f, Macro-F1=%.4f", seed, micro_f1, macro_f1)

    return all_results, feature_cols


# ============================================================================
# Step 6: Compare old vs new
# ============================================================================

def load_old_results():
    """Load existing model results for comparison."""
    old_metrics_path = PROJECT_DIR / "results" / "final_metrics.json"
    if old_metrics_path.exists():
        with open(old_metrics_path) as f:
            return json.load(f)
    return None


def compare_results(new_results, output_dir):
    """Create side-by-side comparison of old vs new results."""
    old = load_old_results()

    # Average new results across seeds
    new_micro_f1s = [r["micro_f1"] for r in new_results]
    new_macro_f1s = [r["macro_f1"] for r in new_results]

    comparison = {
        "old": {
            "n_gt_samples": 623,
            "n_test_samples": 436,
        },
        "new": {
            "n_gt_samples": new_results[0]["n_test"] + new_results[0]["n_train_bench_dev"],
            "n_test_samples": new_results[0]["n_test"],
            "n_train_bench_dev": new_results[0]["n_train_bench_dev"],
            "n_train_prod": new_results[0]["n_train_prod"],
        },
        "metrics": {
            "old_micro_f1": None,
            "new_micro_f1_mean": float(np.mean(new_micro_f1s)),
            "new_micro_f1_std": float(np.std(new_micro_f1s)),
            "old_macro_f1": None,
            "new_macro_f1_mean": float(np.mean(new_macro_f1s)),
            "new_macro_f1_std": float(np.std(new_macro_f1s)),
        },
        "per_dimension": {},
    }

    if old:
        # Extract old metrics
        xgb = old.get("xgboost", old.get("best_model", {}))
        comparison["metrics"]["old_micro_f1"] = xgb.get("micro_f1")
        comparison["metrics"]["old_macro_f1"] = xgb.get("macro_f1")

        old_per_dim = xgb.get("per_label", xgb.get("per_dimension", {}))
        for dim in DIMENSIONS:
            old_dim = old_per_dim.get(dim, {})
            new_dim_f1s = [r["per_dimension"][dim]["f1"] for r in new_results]
            new_dim_supports = [r["per_dimension"][dim]["support"] for r in new_results]
            comparison["per_dimension"][dim] = {
                "old_f1": old_dim.get("f1"),
                "old_support": old_dim.get("support"),
                "new_f1_mean": float(np.mean(new_dim_f1s)),
                "new_f1_std": float(np.std(new_dim_f1s)),
                "new_support": int(new_dim_supports[0]),
                "delta_f1": float(np.mean(new_dim_f1s) - old_dim.get("f1", 0))
                            if old_dim.get("f1") is not None else None,
            }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: OLD (623 GT) vs NEW (623 + boost logs)")
    logger.info("=" * 70)
    logger.info("%-28s %12s %12s %8s", "Metric", "Old", "New (mean±std)", "Delta")
    logger.info("-" * 70)
    old_mi = comparison["metrics"]["old_micro_f1"]
    new_mi = comparison["metrics"]["new_micro_f1_mean"]
    new_mi_s = comparison["metrics"]["new_micro_f1_std"]
    delta_mi = (new_mi - old_mi) if old_mi else None
    logger.info("%-28s %12s %12s %8s",
                "Micro-F1",
                f"{old_mi:.4f}" if old_mi else "N/A",
                f"{new_mi:.4f}±{new_mi_s:.4f}",
                f"{delta_mi:+.4f}" if delta_mi is not None else "N/A")

    old_ma = comparison["metrics"]["old_macro_f1"]
    new_ma = comparison["metrics"]["new_macro_f1_mean"]
    new_ma_s = comparison["metrics"]["new_macro_f1_std"]
    delta_ma = (new_ma - old_ma) if old_ma else None
    logger.info("%-28s %12s %12s %8s",
                "Macro-F1",
                f"{old_ma:.4f}" if old_ma else "N/A",
                f"{new_ma:.4f}±{new_ma_s:.4f}",
                f"{delta_ma:+.4f}" if delta_ma is not None else "N/A")

    logger.info("")
    logger.info("Per-dimension breakdown:")
    logger.info("%-28s %8s %6s %12s %6s %8s", "Dimension", "Old F1", "Old n", "New F1", "New n", "Delta")
    logger.info("-" * 70)
    for dim in DIMENSIONS:
        d = comparison["per_dimension"].get(dim, {})
        old_f1 = d.get("old_f1")
        old_n = d.get("old_support")
        new_f1 = d.get("new_f1_mean")
        new_f1_s = d.get("new_f1_std", 0)
        new_n = d.get("new_support")
        delta = d.get("delta_f1")
        logger.info("%-28s %8s %6s %12s %6s %8s",
                    dim,
                    f"{old_f1:.3f}" if old_f1 is not None else "N/A",
                    str(old_n) if old_n is not None else "N/A",
                    f"{new_f1:.3f}±{new_f1_s:.3f}" if new_f1 is not None else "N/A",
                    str(new_n) if new_n is not None else "N/A",
                    f"{delta:+.3f}" if delta is not None else "N/A")

    logger.info("=" * 70)
    return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Boost experiment — isolated GT expansion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-weight", type=float, default=100.0)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only do feature extraction and splitting")
    args = parser.parse_args()

    log_dir = PROJECT_DIR / "data" / "benchmark_logs" / "ior"

    logger.info("=" * 70)
    logger.info("BOOST EXPERIMENT: Expanding GT with new benchmark logs")
    logger.info("Output: %s", EXPERIMENT_DIR)
    logger.info("=" * 70)

    # ---- Step 1: Map logs to labels ----
    logger.info("\n=== STEP 1: Map Darshan logs to construction-based labels ===")

    mappings_boost = map_darshan_logs_to_labels(
        "17310653", BOOST_JOB_LABELS, log_dir
    )
    mappings_random = map_darshan_logs_to_labels(
        "17309295", RANDOM_BOOST_LABELS, log_dir
    )

    all_mappings = mappings_boost + mappings_random
    logger.info("Total mapped logs: %d (boost=%d, random=%d)",
                len(all_mappings), len(mappings_boost), len(mappings_random))

    # Save mappings for traceability
    with open(EXPERIMENT_DIR / "logs" / "label_mappings.json", "w") as f:
        json.dump([{k: str(v) if isinstance(v, Path) else v
                    for k, v in m.items()} for m in all_mappings], f, indent=2)

    # ---- Step 2: Extract features ----
    logger.info("\n=== STEP 2: Parse and extract features from new logs ===")
    new_features, new_labels = extract_features_from_logs(all_mappings)

    if new_features is None:
        logger.error("No features extracted. Aborting.")
        return

    logger.info("New features shape: %s", new_features.shape)
    logger.info("New label distribution:")
    for dim in DIMENSIONS:
        logger.info("  %s: %d", dim, int(new_labels[dim].sum()))

    # Save new-only data
    new_features.to_parquet(EXPERIMENT_DIR / "new_gt" / "new_features.parquet", index=False)
    new_labels.to_parquet(EXPERIMENT_DIR / "new_gt" / "new_labels.parquet", index=False)

    # ---- Step 3-4: Merge and split ----
    logger.info("\n=== STEP 3-4: Merge with existing GT and re-split ===")
    combined_features, combined_labels, dev_idx, test_idx = merge_and_split(
        new_features, new_labels,
        output_dir=EXPERIMENT_DIR / "new_splits",
        seed=args.seed,
        test_ratio=0.7,
    )

    if args.skip_training:
        logger.info("Skipping training (--skip-training). Done.")
        return

    # ---- Step 5: Train models ----
    logger.info("\n=== STEP 5: Train XGBoost biquality models (5 seeds) ===")
    results, feature_cols = train_models(
        gt_dir=EXPERIMENT_DIR / "new_splits",
        output_model_dir=EXPERIMENT_DIR / "new_models",
        seeds=[42, 123, 456, 789, 1024],
        clean_weight=args.clean_weight,
    )

    # Save raw results
    with open(EXPERIMENT_DIR / "new_evaluation" / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Step 6: Compare ----
    logger.info("\n=== STEP 6: Compare old vs new results ===")
    comparison = compare_results(results, EXPERIMENT_DIR / "comparison")

    logger.info("\nExperiment complete. All results saved to: %s", EXPERIMENT_DIR)
    logger.info("Key files:")
    logger.info("  comparison/comparison.json  — side-by-side old vs new")
    logger.info("  new_evaluation/training_results.json — per-seed results")
    logger.info("  new_splits/ — new GT dataset with splits")
    logger.info("  new_models/ — retrained models (isolated)")


if __name__ == "__main__":
    main()
