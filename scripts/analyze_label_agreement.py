#!/usr/bin/env python3
"""
Label Agreement Analysis: Compare Drishti heuristic labels vs benchmark
construction labels on the SAME benchmark logs.

This is critical for the paper:
1. Quantifies the gap between heuristic and ground-truth labels
2. Shows which dimensions have highest disagreement
3. Provides the "Drishti baseline" evaluation
4. Informs Cleanlab noise correction strategy

Usage:
    python scripts/analyze_label_agreement.py
    python scripts/analyze_label_agreement.py --output results/label_agreement.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix,
)

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


def load_benchmark_data():
    """Load benchmark features and construction labels."""
    feat_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "features.parquet"
    label_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "labels.parquet"

    features = pd.read_parquet(feat_path)
    labels = pd.read_parquet(label_path)
    return features, labels


def apply_drishti_to_benchmark(features):
    """Apply vectorized Drishti rules to benchmark features.

    Returns DataFrame with same 8 dimension columns.
    """
    from src.data.drishti_labeling import compute_drishti_codes, codes_to_labels

    codes = compute_drishti_codes(features)
    labels = codes_to_labels(codes)
    return labels


def compute_agreement(gt_labels, drishti_labels):
    """Compute per-dimension agreement metrics."""
    results = {}

    for dim in DIMENSIONS:
        if dim not in gt_labels.columns or dim not in drishti_labels.columns:
            continue

        y_gt = gt_labels[dim].values.astype(int)
        y_dr = drishti_labels[dim].values.astype(int)

        n_pos_gt = y_gt.sum()
        n_pos_dr = y_dr.sum()

        # Skip if no positive samples in either
        if n_pos_gt == 0 and n_pos_dr == 0:
            results[dim] = {
                "gt_positive": 0,
                "drishti_positive": 0,
                "agreement_rate": 1.0,
                "note": "Both always negative — trivial agreement",
            }
            continue

        # Agreement metrics
        agree = (y_gt == y_dr).mean()

        # Drishti as classifier, GT as ground truth
        f1 = f1_score(y_gt, y_dr, zero_division=0)
        precision = precision_score(y_gt, y_dr, zero_division=0)
        recall = recall_score(y_gt, y_dr, zero_division=0)

        # Cohen's kappa (chance-adjusted agreement)
        kappa = cohen_kappa_score(y_gt, y_dr) if len(set(y_gt)) > 1 else 0.0

        # Confusion matrix
        cm = confusion_matrix(y_gt, y_dr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results[dim] = {
            "gt_positive": int(n_pos_gt),
            "gt_rate": float(n_pos_gt / len(y_gt)),
            "drishti_positive": int(n_pos_dr),
            "drishti_rate": float(n_pos_dr / len(y_dr)),
            "agreement_rate": float(agree),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "kappa": float(kappa),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    # Overall metrics (multi-label)
    y_gt_all = gt_labels[DIMENSIONS].values.astype(int)
    y_dr_all = drishti_labels[DIMENSIONS].values.astype(int)

    results["_overall"] = {
        "micro_f1": float(f1_score(y_gt_all, y_dr_all, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_gt_all, y_dr_all, average="macro", zero_division=0)),
        "hamming_agreement": float(1 - np.mean(y_gt_all != y_dr_all)),
        "n_samples": len(y_gt_all),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze label agreement: Drishti vs benchmark GT")
    parser.add_argument("--output", default="results/label_agreement.json")
    args = parser.parse_args()

    logger.info("Loading benchmark data...")
    features, gt_labels = load_benchmark_data()
    logger.info("Benchmark: %d samples, %d features", len(features), len(features.columns))

    logger.info("Applying Drishti rules to benchmark features...")
    drishti_labels = apply_drishti_to_benchmark(features)
    logger.info("Drishti labels generated: %d samples", len(drishti_labels))

    logger.info("Computing agreement metrics...")
    results = compute_agreement(gt_labels, drishti_labels)

    # Display results
    logger.info("")
    logger.info("=" * 80)
    logger.info("LABEL AGREEMENT: Drishti Heuristic vs Benchmark Ground-Truth")
    logger.info("=" * 80)
    logger.info("")

    header = f"{'Dimension':<28s} {'GT+':<5s} {'Dr+':<5s} {'Agree':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Kappa':>6s}"
    logger.info(header)
    logger.info("-" * len(header))

    for dim in DIMENSIONS:
        if dim not in results:
            continue
        r = results[dim]
        if "note" in r:
            logger.info(f"{dim:<28s} {r['gt_positive']:<5d} {r['drishti_positive']:<5d} {r['agreement_rate']:6.3f}   ({r['note']})")
        else:
            logger.info(f"{dim:<28s} {r['gt_positive']:<5d} {r['drishti_positive']:<5d} {r['agreement_rate']:6.3f} {r['f1']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['kappa']:6.3f}")

    ovr = results["_overall"]
    logger.info("")
    logger.info("Overall: Micro-F1=%.4f  Macro-F1=%.4f  Hamming-Agree=%.4f  N=%d",
                ovr["micro_f1"], ovr["macro_f1"], ovr["hamming_agreement"], ovr["n_samples"])
    logger.info("=" * 80)

    # Save results
    output_path = PROJECT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
