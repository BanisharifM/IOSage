"""
GT-only XGBoost training (single seed) for the Table III baseline row.

Trains on the 201 benchmark dev samples only (no heuristic production data, no
biquality), then evaluates on the 488 benchmark test set. Designed to be run as
5 independent SLURM jobs (one per seed) for parallel wall-time, then aggregated
by aggregate_5seeds.py.

This script is a slim copy of results/boost_experiment/scripts/train_biquality_boost.py
with the production-data path removed and the output format switched to per-seed
JSON instead of pickle. No re-implementation: it reuses the same data loaders,
same hyperparameter source (configs/training.yaml), same evaluation function.

Usage:
    python train_gt_only_single_seed.py --seed 42 --output-dir output/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Import the canonical functions from the existing biquality script.
PROJECT_DIR = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
sys.path.insert(0, str(PROJECT_DIR / "results" / "boost_experiment" / "scripts"))

from train_biquality_boost import (  # noqa: E402
    DIMENSIONS,
    compute_scale_pos_weight,
    evaluate_on_gt_test,
    load_benchmark_data,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def train_gt_only(X_bench_dev, y_bench_dev, config, seed=42):
    """Train XGBoost per-label on the benchmark dev set only (no heuristic data)."""
    from xgboost import XGBClassifier

    spw = compute_scale_pos_weight(y_bench_dev, max_weight=100.0)
    models = {}
    params = config["models"]["xgboost"]["params"].copy()

    logger.info(
        "  GT-only training: %d benchmark dev samples, %d features",
        len(X_bench_dev),
        X_bench_dev.shape[1],
    )

    for i, dim in enumerate(DIMENSIONS):
        clf = XGBClassifier(
            **params,
            scale_pos_weight=spw[i],
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_bench_dev, y_bench_dev[:, i], verbose=False)
        models[dim] = clf

    return models


def main():
    parser = argparse.ArgumentParser(
        description="GT-only XGBoost training, single seed (for SLURM parallel runs)"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for this single run")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save the per-seed JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Load the canonical 157 feature column names from a pre-saved JSON.
    # This avoids loading production data (which triggers a pickle.load of
    # numpy-2.x-saved splits that fails under numpy 1.x on compute nodes).
    feature_cols_path = Path(__file__).parent / "feature_cols.json"
    with open(feature_cols_path) as f:
        feature_cols = json.load(f)
    logger.info("Feature columns: %d (from %s)", len(feature_cols), feature_cols_path.name)

    logger.info("Loading benchmark data...")
    X_bench, y_bench, dev_idx, test_idx, _ = load_benchmark_data(config, feature_cols)
    X_bench_dev, y_bench_dev = X_bench[dev_idx], y_bench[dev_idx]
    X_bench_test, y_bench_test = X_bench[test_idx], y_bench[test_idx]
    logger.info("Benchmark: %d dev, %d test", len(dev_idx), len(test_idx))

    # Log GT label distribution on dev (the training set for this script)
    logger.info("Benchmark DEV (training) label distribution:")
    for i, dim in enumerate(DIMENSIONS):
        n = int(y_bench_dev[:, i].sum())
        logger.info("  %-28s %4d (%5.1f%%)", dim, n, 100 * n / len(y_bench_dev))

    logger.info("")
    logger.info("=" * 75)
    logger.info("GT-only XGBoost training, seed=%d", args.seed)
    logger.info("=" * 75)

    t0 = time.time()
    models = train_gt_only(X_bench_dev, y_bench_dev, config, seed=args.seed)
    train_time = time.time() - t0
    logger.info("Training completed in %.1fs", train_time)

    metrics, y_pred, y_prob = evaluate_on_gt_test(models, X_bench_test, y_bench_test)

    logger.info("")
    logger.info("Results (GT-only XGBoost, seed=%d):", args.seed)
    logger.info("  Micro-F1:   %.4f", metrics["micro_f1"])
    logger.info("  Macro-F1:   %.4f", metrics["macro_f1"])
    logger.info("  Hamming:    %.4f", metrics["hamming_loss"])
    logger.info("  Subset Acc: %.4f", metrics["subset_accuracy"])

    # Build the per-seed record (same shape as iosage_xgboost_5seeds_full.json
    # per_seed entry, so the aggregator can read both with the same code).
    per_seed_record = {
        "seed": args.seed,
        "micro_f1": float(metrics["micro_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "hamming_loss": float(metrics["hamming_loss"]),
        "subset_accuracy": float(metrics["subset_accuracy"]),
        "micro_f1_ci": [float(metrics["micro_f1_ci"][0]), float(metrics["micro_f1_ci"][1])],
        "macro_f1_ci": [float(metrics["macro_f1_ci"][0]), float(metrics["macro_f1_ci"][1])],
        "per_label": {
            dim: {k: float(v) if k != "support" else int(v) for k, v in m.items()}
            for dim, m in metrics["per_label"].items()
        },
        "train_time_seconds": float(train_time),
    }

    config_record = {
        "model_family": "XGBoost",
        "training_mode": "GT-only (no heuristic, no biquality)",
        "n_heuristic_samples": 0,
        "n_gt_samples": int(len(X_bench_dev)),
        "n_features": int(X_bench_dev.shape[1]),
        "threshold_for_metrics": 0.5,
        "n_test_samples": int(len(X_bench_test)),
        "n_dimensions": len(DIMENSIONS),
        "seed": args.seed,
    }

    output_path = output_dir / f"gt_only_xgboost_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(
            {"training_config": config_record, "per_seed": {str(args.seed): per_seed_record}},
            f,
            indent=2,
        )
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
