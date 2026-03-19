#!/usr/bin/env python3
"""
Prepare Phase 2 training data: split benchmark GT and combine with production.

1. Split benchmark data using iterative stratification (30% dev / 70% test)
2. Create combined training set (production heuristic + benchmark dev)
3. Apply source-aware weighting
4. Save all splits for reproducibility

References:
    - Sechidis et al. (ECML-PKDD 2011): iterative stratification for multi-label
    - Ratner et al. (VLDB 2018): Snorkel paradigm — train on weak, test on gold
    - Zhang et al. (NeurIPS 2021): WRENCH benchmark splits

Usage:
    python scripts/prepare_phase2_data.py
    python scripts/prepare_phase2_data.py --benchmark-test-ratio 0.7 --seed 42
"""

import argparse
import json
import logging
import pickle
import sys
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

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


def load_config():
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        return yaml.safe_load(f)


def load_production_data(config):
    """Load production features, labels, and existing splits."""
    paths = config["paths"]
    features = pd.read_parquet(PROJECT_DIR / paths["production_features"])
    labels = pd.read_parquet(PROJECT_DIR / paths["production_labels"])

    # Align
    labels = labels.set_index("_jobid")
    features = features.set_index("_jobid")
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # Load existing temporal split
    split_path = PROJECT_DIR / paths["production_splits"]
    with open(split_path, "rb") as f:
        splits = pickle.load(f)

    return features, labels, splits


def load_benchmark_data(config):
    """Load benchmark features and construction labels."""
    paths = config["paths"]
    features = pd.read_parquet(PROJECT_DIR / paths["benchmark_features"])
    labels = pd.read_parquet(PROJECT_DIR / paths["benchmark_labels"])
    return features, labels


def split_benchmark(features, labels, test_ratio=0.7, seed=42):
    """Split benchmark data using iterative stratification.

    Args:
        test_ratio: fraction for test (evaluation). Default 0.7.
        seed: random seed for reproducibility.

    Returns:
        dev_idx, test_idx: arrays of indices
    """
    y = labels[DIMENSIONS].values.astype(int)

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )

    for dev_idx, test_idx in msss.split(features.values, y):
        pass  # Only one split

    # Log distribution
    y_dev = y[dev_idx]
    y_test = y[test_idx]

    logger.info("Benchmark split: %d dev (%.0f%%), %d test (%.0f%%)",
                len(dev_idx), 100 * len(dev_idx) / len(y),
                len(test_idx), 100 * len(test_idx) / len(y))

    logger.info("")
    logger.info("Label distribution (iterative stratification):")
    header = f"  {'Dimension':<28s} {'Dev':>5s} {'Test':>5s} {'Dev%':>6s} {'Test%':>6s}"
    logger.info(header)
    for i, dim in enumerate(DIMENSIONS):
        nd = y_dev[:, i].sum()
        nt = y_test[:, i].sum()
        dp = 100 * nd / max(nd + nt, 1)
        tp = 100 * nt / max(nd + nt, 1)
        logger.info(f"  {dim:<28s} {nd:5d} {nt:5d} {dp:5.1f}% {tp:5.1f}%")

    return dev_idx, test_idx


def prepare_combined_training(prod_features, prod_labels, prod_splits,
                               bench_features, bench_labels, bench_dev_idx,
                               config):
    """Create combined training set with source-aware weighting.

    Production train samples get weight = heuristic_weight (default 0.8)
    Benchmark dev samples get weight = ground_truth_weight (default 1.0)
    """
    exclude = set(config.get("exclude_features", []))
    for col in prod_features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_features.columns if c not in exclude]

    # Production training data
    train_idx = prod_splits["train_idx"]
    X_prod_train = prod_features.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_prod_train = prod_labels.iloc[train_idx][DIMENSIONS].values.astype(np.float32)

    # Benchmark dev data (same feature columns, fill missing with 0)
    bench_dev = bench_features.iloc[bench_dev_idx]
    X_bench_dev_cols = []
    for col in feature_cols:
        if col in bench_dev.columns:
            X_bench_dev_cols.append(bench_dev[col].values)
        else:
            X_bench_dev_cols.append(np.zeros(len(bench_dev)))
    X_bench_dev = np.column_stack(X_bench_dev_cols).astype(np.float32)
    y_bench_dev = bench_labels.iloc[bench_dev_idx][DIMENSIONS].values.astype(np.float32)

    # Combine
    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])

    # Source-aware weights
    gt_weight = config.get("training_data", {}).get("ground_truth_weight", 1.0)
    heur_weight = config.get("training_data", {}).get("heuristic_weight", 0.8)

    weights = np.concatenate([
        np.full(len(X_prod_train), heur_weight),
        np.full(len(X_bench_dev), gt_weight),
    ])

    source = np.array(
        ["production"] * len(X_prod_train) + ["benchmark"] * len(X_bench_dev)
    )

    logger.info("")
    logger.info("Combined training set:")
    logger.info("  Production: %d samples (weight=%.1f)", len(X_prod_train), heur_weight)
    logger.info("  Benchmark:  %d samples (weight=%.1f)", len(X_bench_dev), gt_weight)
    logger.info("  Total:      %d samples", len(X_combined))

    return X_combined, y_combined, weights, source, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase 2 training data")
    parser.add_argument("--benchmark-test-ratio", type=float, default=0.7,
                        help="Fraction of benchmark data for test (default 0.7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed/benchmark")
    args = parser.parse_args()

    config = load_config()

    # Load data
    logger.info("Loading production data...")
    prod_features, prod_labels, prod_splits = load_production_data(config)
    logger.info("Production: %d samples", len(prod_features))

    logger.info("Loading benchmark data...")
    bench_features, bench_labels = load_benchmark_data(config)
    logger.info("Benchmark: %d samples", len(bench_features))

    # Split benchmark
    logger.info("")
    logger.info("Splitting benchmark data (iterative stratification)...")
    dev_idx, test_idx = split_benchmark(
        bench_features, bench_labels,
        test_ratio=args.benchmark_test_ratio,
        seed=args.seed,
    )

    # Save benchmark split indices
    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir / "split_indices.pkl"
    with open(split_path, "wb") as f:
        pickle.dump({"dev_idx": dev_idx, "test_idx": test_idx}, f)
    logger.info("Saved benchmark splits to %s", split_path)

    # Prepare combined training set
    logger.info("")
    logger.info("Preparing combined training set...")
    X_combined, y_combined, weights, source, feature_cols = prepare_combined_training(
        prod_features, prod_labels, prod_splits,
        bench_features, bench_labels, dev_idx,
        config,
    )

    # Save combined data
    combined_path = output_dir / "combined_train.npz"
    np.savez_compressed(
        combined_path,
        X=X_combined,
        y=y_combined,
        weights=weights,
        source=source,
        feature_cols=np.array(feature_cols),
    )
    logger.info("Saved combined training data to %s", combined_path)

    # Save benchmark test set separately
    bench_test_features = bench_features.iloc[test_idx]
    bench_test_labels = bench_labels.iloc[test_idx]
    bench_test_features.to_parquet(output_dir / "test_features.parquet", index=False)
    bench_test_labels.to_parquet(output_dir / "test_labels.parquet", index=False)
    logger.info("Saved benchmark test set: %d samples", len(test_idx))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2 DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Benchmark dev:  %d samples (for fine-tuning)", len(dev_idx))
    logger.info("Benchmark test: %d samples (for evaluation)", len(test_idx))
    logger.info("Combined train: %d samples (production + benchmark dev)", len(X_combined))
    logger.info("")
    logger.info("Files saved to %s:", output_dir)
    logger.info("  split_indices.pkl       — benchmark dev/test split")
    logger.info("  combined_train.npz      — combined training data with weights")
    logger.info("  test_features.parquet   — benchmark test features")
    logger.info("  test_labels.parquet     — benchmark test labels")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
