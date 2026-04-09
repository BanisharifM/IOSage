"""
Aggregate the 5 per-seed JSONs from train_gt_only_single_seed.py into a single
JSON file with the same shape as iosage_xgboost_5seeds_full.json (training_config
+ per_seed dict + aggregate_5_seed dict). Run after all 5 SLURM jobs finish.

Usage:
    python aggregate_5seeds.py
"""

import json
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
SEEDS = [42, 123, 456, 789, 1024]


def main():
    per_seed = {}
    config = None

    for seed in SEEDS:
        path = OUTPUT_DIR / f"gt_only_xgboost_seed{seed}.json"
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            d = json.load(f)
        per_seed[str(seed)] = d["per_seed"][str(seed)]
        if config is None:
            config = d["training_config"]
            del config["seed"]
            config["seeds"] = SEEDS

    def mean_std(key):
        vals = [per_seed[str(s)][key] for s in SEEDS]
        return float(np.mean(vals)), float(np.std(vals))

    aggregate = {}
    for k in ["micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"]:
        m, s = mean_std(k)
        aggregate[f"{k}_mean"] = m
        aggregate[f"{k}_std"] = s

    # Per-label aggregation
    dims = list(per_seed[str(SEEDS[0])]["per_label"].keys())
    per_label_agg = {}
    for dim in dims:
        per_label_agg[dim] = {}
        for metric in ["f1", "precision", "recall"]:
            vals = [per_seed[str(s)]["per_label"][dim][metric] for s in SEEDS]
            per_label_agg[dim][f"{metric}_mean"] = float(np.mean(vals))
            per_label_agg[dim][f"{metric}_std"] = float(np.std(vals))
        per_label_agg[dim]["support"] = per_seed[str(SEEDS[0])]["per_label"][dim]["support"]
    aggregate["per_label"] = per_label_agg

    out = {
        "training_config": config,
        "per_seed": per_seed,
        "aggregate_5_seed": aggregate,
    }

    out_path = OUTPUT_DIR.parent / "gt_only_xgboost_5seeds_full.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 75)
    print("GT-only XGBoost 5-seed aggregate")
    print("=" * 75)
    print(f"Micro-F1:   {aggregate['micro_f1_mean']:.4f} +/- {aggregate['micro_f1_std']:.4f}")
    print(f"Macro-F1:   {aggregate['macro_f1_mean']:.4f} +/- {aggregate['macro_f1_std']:.4f}")
    print(f"Hamming:    {aggregate['hamming_loss_mean']:.4f} +/- {aggregate['hamming_loss_std']:.4f}")
    print(f"Subset Acc: {aggregate['subset_accuracy_mean']:.4f} +/- {aggregate['subset_accuracy_std']:.4f}")
    print(f"\nSaved aggregate: {out_path}")


if __name__ == "__main__":
    main()
