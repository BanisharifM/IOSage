#!/usr/bin/env python3
"""
Extract features and labels from benchmark Darshan logs.

Produces ground-truth feature vectors and labels consistent with the
production pipeline (same extract_raw_features, same column order).

For IOR/mdtest (compiled MPI): one aggregated .darshan per job → parse directly.
For DLIO/custom (Python + LD_PRELOAD): per-rank .darshan files → aggregate via
parse_benchmark_job() using the same 7 Darshan aggregation rules as MPI_Finalize.

Output:
    data/processed/ground_truth_features.parquet  — same columns as raw_features.parquet
    data/processed/ground_truth_labels.parquet     — 8 binary label dimensions + metadata

Usage:
    python scripts/extract_benchmark_features.py
    python scripts/extract_benchmark_features.py --output-dir data/processed
    python scripts/extract_benchmark_features.py --bench-type ior  # single benchmark
"""

import argparse
import glob
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parse_darshan import parse_darshan_log, parse_benchmark_job
from src.data.feature_extraction import (
    extract_raw_features,
    get_raw_feature_names,
    get_info_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_DIR / "data" / "benchmark_logs"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "data" / "benchmark_results"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

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

# Benchmark types that produce per-rank logs (Python + LD_PRELOAD)
PER_RANK_BENCHMARKS = {"dlio", "custom"}

# Benchmark types that produce aggregated logs (compiled MPI)
AGGREGATED_BENCHMARKS = {"ior", "mdtest"}


# ---------------------------------------------------------------------------
# Label extraction from SLURM output
# ---------------------------------------------------------------------------

def find_job_label(job_id, results_dir):
    """Find the expected label string from SLURM stdout by matching job ID."""
    for out_file in glob.glob(os.path.join(results_dir, f"*_{job_id}.out")):
        with open(out_file) as f:
            for line in f:
                if line.strip().startswith("Label:"):
                    return line.strip().split("Label:")[-1].strip()
    return None


def find_job_scenario(job_id, results_dir):
    """Extract scenario name from SLURM output filename."""
    for out_file in glob.glob(os.path.join(results_dir, f"*_{job_id}.out")):
        basename = os.path.basename(out_file)
        # Remove _JOBID.out suffix to get scenario name
        scenario = re.sub(r"_\d+\.out$", "", basename)
        return scenario
    return None


def parse_label_string(label_str):
    """Parse 'access_granularity=1,interface_choice=1' into label dict."""
    labels = {dim: 0 for dim in DIMENSION_NAMES}
    if not label_str:
        return labels
    for part in label_str.split(","):
        if "=" in part:
            key, val = part.strip().split("=", 1)
            key = key.strip()
            if key in labels:
                labels[key] = int(val)
    # Set healthy: 1 if no bottleneck dimensions active
    bottleneck_sum = sum(v for k, v in labels.items() if k != "healthy")
    if bottleneck_sum == 0:
        labels["healthy"] = 1
    else:
        labels["healthy"] = 0
    return labels


# ---------------------------------------------------------------------------
# Per-rank log grouping
# ---------------------------------------------------------------------------

def group_logs_by_job(log_dir, bench_type):
    """Group per-rank .darshan files by PBS/SLURM job ID.

    Returns dict: {job_id: [list of .darshan file paths]}
    Filters out probe logs (lscpu, uname) for DLIO.
    """
    darshan_files = sorted(glob.glob(os.path.join(log_dir, "*.darshan")))
    jobs = defaultdict(list)

    for fpath in darshan_files:
        basename = os.path.basename(fpath)

        # Skip DLIO probe logs
        if bench_type == "dlio" and ("_lscpu_" in basename or "_uname_" in basename):
            continue

        # Extract job ID: pattern is _id{JOBID}-{RANK}_
        match = re.search(r"_id(\d+)-", basename)
        if match:
            jobs[match.group(1)].append(fpath)
        else:
            # Fallback: treat as single-file job
            jobs[basename].append(fpath)

    return dict(jobs)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_aggregated_benchmark(bench_type, log_dir, results_dir):
    """Extract features from aggregated benchmark logs (IOR, mdtest).

    Each .darshan file is one job → parse directly.
    """
    darshan_files = sorted(glob.glob(os.path.join(log_dir, "*.darshan")))
    if not darshan_files:
        logger.warning("No logs found in %s", log_dir)
        return [], []

    logger.info("Processing %d %s logs (aggregated)...", len(darshan_files), bench_type)

    feature_rows = []
    label_rows = []
    n_ok = 0
    n_fail = 0

    for fpath in darshan_files:
        basename = os.path.basename(fpath)

        # Extract job ID for label lookup
        match = re.search(r"_id(\d+)", basename)
        job_id = match.group(1) if match else None

        # Parse log
        parsed = parse_darshan_log(fpath)
        if parsed is None:
            logger.warning("  Failed to parse: %s", basename)
            n_fail += 1
            continue

        # Extract features
        features = extract_raw_features(parsed)

        # Find label from SLURM output
        label_str = find_job_label(job_id, results_dir) if job_id else None
        labels = parse_label_string(label_str)
        scenario = find_job_scenario(job_id, results_dir) if job_id else None

        # Add source path for traceability
        features["_source_path"] = fpath
        features["_benchmark"] = bench_type
        features["_scenario"] = scenario or ""
        features["_ground_truth_job_id"] = job_id or ""

        feature_rows.append(features)

        label_row = {
            "job_id": job_id or "",
            "benchmark": bench_type,
            "scenario": scenario or "",
            "n_ranks": features.get("nprocs", 1),
            "n_darshan_files": 1,
            "label_source": "construction" if label_str else "unknown",
        }
        label_row.update(labels)
        label_rows.append(label_row)
        n_ok += 1

    logger.info("  %s: %d extracted, %d failed", bench_type, n_ok, n_fail)
    return feature_rows, label_rows


def extract_perrank_benchmark(bench_type, log_dir, results_dir):
    """Extract features from per-rank benchmark logs (DLIO, custom).

    Groups per-rank .darshan files by job ID, aggregates via
    parse_benchmark_job(), then extracts features.
    """
    job_groups = group_logs_by_job(log_dir, bench_type)
    if not job_groups:
        logger.warning("No jobs found in %s", log_dir)
        return [], []

    n_jobs = len(job_groups)
    n_files = sum(len(v) for v in job_groups.values())
    logger.info(
        "Processing %d %s jobs (%d per-rank files, aggregating)...",
        n_jobs, bench_type, n_files,
    )

    feature_rows = []
    label_rows = []
    n_ok = 0
    n_fail = 0

    for job_id, rank_files in sorted(job_groups.items()):
        # Aggregate per-rank logs into one job-level result
        parsed = parse_benchmark_job(rank_files)
        if parsed is None:
            logger.warning("  Failed to aggregate job %s (%d files)", job_id, len(rank_files))
            n_fail += 1
            continue

        # Extract features (identical pipeline as production)
        features = extract_raw_features(parsed)

        # Find label from SLURM output
        label_str = find_job_label(job_id, results_dir)
        labels = parse_label_string(label_str)
        scenario = find_job_scenario(job_id, results_dir)

        # Add source metadata
        features["_source_path"] = rank_files[0]  # first rank file as reference
        features["_benchmark"] = bench_type
        features["_scenario"] = scenario or ""
        features["_ground_truth_job_id"] = job_id

        feature_rows.append(features)

        label_row = {
            "job_id": job_id,
            "benchmark": bench_type,
            "scenario": scenario or "",
            "n_ranks": len(rank_files),
            "n_darshan_files": len(rank_files),
            "label_source": "construction" if label_str else "unknown",
        }
        label_row.update(labels)
        label_rows.append(label_row)
        n_ok += 1

    logger.info("  %s: %d jobs extracted, %d failed", bench_type, n_ok, n_fail)
    return feature_rows, label_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract features and labels from benchmark Darshan logs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help="Root directory with benchmark_logs/{ior,mdtest,dlio,custom}/",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Root directory with benchmark_results/{ior,mdtest,dlio,custom}/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--bench-type",
        type=str,
        choices=["all", "ior", "mdtest", "dlio", "custom"],
        default="all",
        help="Which benchmark type to process",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_types = (
        ["ior", "mdtest", "dlio", "custom"]
        if args.bench_type == "all"
        else [args.bench_type]
    )

    all_features = []
    all_labels = []

    for bt in bench_types:
        bt_log_dir = log_dir / bt
        bt_results_dir = results_dir / bt

        if not bt_log_dir.exists():
            logger.warning("Log directory not found: %s", bt_log_dir)
            continue

        if bt in AGGREGATED_BENCHMARKS:
            feats, labs = extract_aggregated_benchmark(
                bt, str(bt_log_dir), str(bt_results_dir)
            )
        elif bt in PER_RANK_BENCHMARKS:
            feats, labs = extract_perrank_benchmark(
                bt, str(bt_log_dir), str(bt_results_dir)
            )
        else:
            logger.warning("Unknown benchmark type: %s", bt)
            continue

        all_features.extend(feats)
        all_labels.extend(labs)

    if not all_features:
        logger.error("No features extracted. Check log directories.")
        sys.exit(1)

    # Build DataFrames
    features_df = pd.DataFrame(all_features)
    labels_df = pd.DataFrame(all_labels)

    # Ensure consistent column order matching production pipeline
    raw_feature_cols = get_raw_feature_names()
    info_cols = get_info_columns()
    extra_cols = ["_source_path", "_benchmark", "_scenario", "_ground_truth_job_id"]

    # Reorder feature columns: raw features + info + extras
    ordered_cols = []
    for col in raw_feature_cols + info_cols + extra_cols:
        if col in features_df.columns:
            ordered_cols.append(col)

    # Add any columns we missed (shouldn't happen, but safety)
    for col in features_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    features_df = features_df[ordered_cols]

    # Ensure label column order
    label_meta_cols = ["job_id", "benchmark", "scenario", "n_ranks", "n_darshan_files", "label_source"]
    label_ordered = label_meta_cols + DIMENSION_NAMES
    labels_df = labels_df[label_ordered]

    # Save
    feat_path = output_dir / "ground_truth_features.parquet"
    label_path = output_dir / "ground_truth_labels.parquet"
    features_df.to_parquet(feat_path, index=False)
    labels_df.to_parquet(label_path, index=False)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GROUND-TRUTH EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total jobs: %d", len(features_df))
    logger.info("Features shape: %s", features_df.shape)
    logger.info("Labels shape: %s", labels_df.shape)
    logger.info("")

    # Per-benchmark breakdown
    for bt in bench_types:
        bt_mask = labels_df["benchmark"] == bt
        n = bt_mask.sum()
        if n > 0:
            label_dist = {
                dim: int(labels_df.loc[bt_mask, dim].sum())
                for dim in DIMENSION_NAMES
            }
            active = {k: v for k, v in label_dist.items() if v > 0}
            logger.info("  %s: %d jobs, labels: %s", bt, n, active)

    logger.info("")
    logger.info("Label distribution (all benchmarks):")
    for dim in DIMENSION_NAMES:
        n_pos = int(labels_df[dim].sum())
        pct = n_pos / len(labels_df) * 100
        logger.info("  %-25s %4d (%5.1f%%)", dim, n_pos, pct)

    n_unknown = (labels_df["label_source"] == "unknown").sum()
    if n_unknown > 0:
        logger.warning(
            "%d jobs have unknown labels (no SLURM .out match)", n_unknown
        )

    logger.info("")
    logger.info("Saved: %s", feat_path)
    logger.info("Saved: %s", label_path)

    # Consistency check: verify column overlap with production raw_features
    prod_path = output_dir / "raw_features.parquet"
    if prod_path.exists():
        prod_cols = set(pd.read_parquet(prod_path, columns=[]).columns)
        gt_cols = set(features_df.columns) - {"_source_path", "_benchmark", "_scenario", "_ground_truth_job_id"}
        missing = prod_cols - gt_cols
        extra = gt_cols - prod_cols
        if missing:
            logger.warning("Columns in production but missing in ground-truth: %s", missing)
        if extra:
            logger.info("Extra columns in ground-truth (metadata): %s", extra)
        overlap = prod_cols & gt_cols
        logger.info("Column overlap with production: %d/%d", len(overlap), len(prod_cols))
    else:
        logger.info("Production raw_features.parquet not found; skipping consistency check")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
