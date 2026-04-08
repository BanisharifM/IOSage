"""
Run WisIO baseline on benchmark Darshan logs and compare against ground-truth labels.

WisIO detects 6 rule-based bottlenecks:
  - excessive_metadata_access
  - operation_imbalance
  - random_operations
  - size_imbalance
  - small_reads
  - small_writes

We map these to our 8-dimension taxonomy:
  - access_granularity      <- small_reads OR small_writes
  - metadata_intensity      <- excessive_metadata_access
  - parallelism_efficiency  <- operation_imbalance OR size_imbalance (approximate)
  - access_pattern          <- random_operations
  - interface_choice        <- (no WisIO mapping)
  - file_strategy           <- (no WisIO mapping)
  - throughput_utilization  <- (no WisIO mapping)
  - healthy                 <- none of the above detected

Usage:
    PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH \
    /projects/bdau/envs/sc2026/bin/python scripts/run_wisio_baseline.py
"""

import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")
os.environ["DASK_DISTRIBUTED__LOGGING__DISTRIBUTED"] = "error"
os.environ["DASK_LOGGING__DISTRIBUTED"] = "error"
logging.disable(logging.WARNING)

# ---------- configuration ----------
# Run WisIO on the NEW boost-experiment 488-sample test set
# (instead of the old 436-sample data/processed/benchmark/labels.parquet)
PROJECT_ROOT = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
BENCHMARK_LOGS_DIR = PROJECT_ROOT / "data" / "benchmark_logs"
LABELS_PATH = PROJECT_ROOT / "results" / "boost_experiment" / "new_splits" / "test_labels.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "wisio_baseline_full488"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WISIO_RULES = [
    "excessive_metadata_access",
    "operation_imbalance",
    "random_operations",
    "size_imbalance",
    "small_reads",
    "small_writes",
]

WISIO_TO_TAXONOMY = {
    "access_granularity": ["small_reads", "small_writes"],
    "metadata_intensity": ["excessive_metadata_access"],
    "parallelism_efficiency": ["operation_imbalance", "size_imbalance"],
    "access_pattern": ["random_operations"],
    "interface_choice": [],
    "file_strategy": [],
    "throughput_utilization": [],
}

TAXONOMY_DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
]


def find_darshan_file(job_id, benchmark):
    """Find the darshan file for a given job_id in benchmark_logs."""
    bench_dir = BENCHMARK_LOGS_DIR / benchmark
    if not bench_dir.exists():
        return None
    matches = list(bench_dir.glob(f"*id{job_id}*"))
    return str(matches[0]) if matches else None


def wisio_rules_to_taxonomy(rule_flags):
    """Convert WisIO rule flags to our 8-dimension taxonomy predictions."""
    preds = {}
    for dim, rules in WISIO_TO_TAXONOMY.items():
        preds[dim] = int(any(rule_flags.get(r, False) for r in rules))
    any_bottleneck = any(preds[d] for d in TAXONOMY_DIMS)
    preds["healthy"] = 0 if any_bottleneck else 1
    return preds


def compute_metrics(y_true, y_pred, dim_names):
    """Compute per-dimension and aggregate metrics."""
    results = {}
    for i, dim in enumerate(dim_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        results[dim] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support_positive": int(yt.sum()),
            "support_negative": int((1 - yt).sum()),
            "predicted_positive": int(yp.sum()),
        }

    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="binary", zero_division=0
    )

    f1_scores = [results[d]["f1"] for d in dim_names]
    f1_macro = float(np.mean(f1_scores))

    mapped_dims = [d for d in dim_names if d != "healthy" and WISIO_TO_TAXONOMY.get(d, [])]
    if mapped_dims:
        mapped_indices = [dim_names.index(d) for d in mapped_dims]
        y_true_mapped = y_true[:, mapped_indices].ravel()
        y_pred_mapped = y_pred[:, mapped_indices].ravel()
        _, _, f1_mapped_micro, _ = precision_recall_fscore_support(
            y_true_mapped, y_pred_mapped, average="binary", zero_division=0
        )
        f1_mapped_scores = [results[d]["f1"] for d in mapped_dims]
        f1_mapped_macro = float(np.mean(f1_mapped_scores))
    else:
        f1_mapped_micro = 0.0
        f1_mapped_macro = 0.0

    # Multi-label metrics needed for tab_baselines:
    # Hamming loss = fraction of incorrect individual labels
    # Subset accuracy = exact-match of the full label vector
    h_loss = float(hamming_loss(y_true, y_pred))
    subset_acc = float(accuracy_score(y_true, y_pred))

    results["_aggregate"] = {
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_f1": f1_macro,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc,
        "mapped_micro_f1": float(f1_mapped_micro),
        "mapped_macro_f1": f1_mapped_macro,
        "n_samples": int(y_true.shape[0]),
        "n_dimensions": int(y_true.shape[1]),
        "mapped_dimensions": mapped_dims,
        "unmapped_dimensions": [
            d for d in dim_names if d != "healthy" and d not in mapped_dims
        ],
    }
    return results


def main():
    print("=" * 70, flush=True)
    print("WisIO Baseline Evaluation on Benchmark Ground-Truth Logs", flush=True)
    print("=" * 70, flush=True)

    # Load ground-truth labels
    labels_df = pd.read_parquet(LABELS_PATH)
    print(f"\nLoaded {len(labels_df)} ground-truth labels", flush=True)
    print(f"Unique job_ids: {labels_df['job_id'].nunique()}", flush=True)
    print(f"Benchmarks: {labels_df['benchmark'].value_counts().to_dict()}", flush=True)

    # Map job_ids to darshan files
    file_map = {}
    for _, row in labels_df.iterrows():
        jid = row["job_id"]
        if jid not in file_map:
            fpath = find_darshan_file(jid, row["benchmark"])
            if fpath:
                file_map[jid] = fpath
    print(f"Found darshan files for {len(file_map)}/{labels_df['job_id'].nunique()} unique jobs", flush=True)

    # Initialize Dask
    from dask.distributed import LocalCluster, Client
    from wisio.darshan import DarshanAnalyzer

    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, memory_limit="4GB", silence_logs=50
    )
    client = Client(cluster)
    print(f"Dask client started", flush=True)

    # Run WisIO: process unique job_ids to avoid redundant work
    wisio_cache = {}  # job_id -> rule_flags
    errors = []
    t_start = time.time()
    unique_jobs = list(file_map.keys())

    for idx, job_id in enumerate(unique_jobs):
        fpath = file_map[job_id]
        try:
            analyzer = DarshanAnalyzer(
                checkpoint=False, checkpoint_dir="",
                bottleneck_dir="/tmp/wisio_baseline_bot", verbose=False,
            )
            result = analyzer.analyze_trace(
                trace_path=fpath,
                percentile=0.9,
                view_types=["file_name", "proc_name"],
                metrics=["iops"],
                exclude_bottlenecks=[],
                exclude_characteristics=[],
            )
            if result._bottlenecks is not None:
                bot_df = result._bottlenecks.compute()
                rule_flags = {}
                for rule in WISIO_RULES:
                    rule_flags[rule] = bool(bot_df[rule].any()) if rule in bot_df.columns else False
            else:
                rule_flags = {r: False for r in WISIO_RULES}

            wisio_cache[job_id] = rule_flags

        except Exception as e:
            errors.append({"job_id": int(job_id), "file": fpath, "error": str(e)})
            wisio_cache[job_id] = {r: False for r in WISIO_RULES}
            if len(errors) <= 10:
                print(f"  ERROR {job_id}: {e}", flush=True)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(unique_jobs) - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx+1}/{len(unique_jobs)}] "
                f"{elapsed:.0f}s elapsed, {rate:.2f} jobs/s, "
                f"ETA: {eta:.0f}s | errors so far: {len(errors)}",
                flush=True,
            )

    elapsed_total = time.time() - t_start
    print(f"\nProcessed {len(wisio_cache)} unique jobs, {len(errors)} errors in {elapsed_total:.1f}s", flush=True)

    client.close()
    cluster.close()

    # Build results for all 623 label rows
    all_results = []
    for _, row in labels_df.iterrows():
        job_id = row["job_id"]
        if job_id not in wisio_cache:
            continue
        rule_flags = wisio_cache[job_id]
        taxonomy_preds = wisio_rules_to_taxonomy(rule_flags)

        record = {
            "job_id": job_id,
            "benchmark": row["benchmark"],
            "scenario": row["scenario"],
        }
        for dim in TAXONOMY_DIMS + ["healthy"]:
            record[f"gt_{dim}"] = int(row[dim])
        for rule in WISIO_RULES:
            record[f"wisio_{rule}"] = int(rule_flags[rule])
        for dim in TAXONOMY_DIMS + ["healthy"]:
            record[f"pred_{dim}"] = taxonomy_preds[dim]
        all_results.append(record)

    if not all_results:
        print("No results to evaluate!")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(OUTPUT_DIR / "wisio_predictions.parquet", index=False)
    results_df.to_csv(OUTPUT_DIR / "wisio_predictions.csv", index=False)
    print(f"\nSaved predictions ({len(results_df)} rows) to {OUTPUT_DIR}", flush=True)

    # Evaluate
    dim_names = TAXONOMY_DIMS + ["healthy"]
    y_true = results_df[[f"gt_{d}" for d in dim_names]].values
    y_pred = results_df[[f"pred_{d}" for d in dim_names]].values

    metrics = compute_metrics(y_true, y_pred, dim_names)

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("=" * 70, flush=True)

    header = f"{'Dimension':<28} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Sup(+)':>7} {'Pred(+)':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for dim in dim_names:
        m = metrics[dim]
        mapped = "*" if WISIO_TO_TAXONOMY.get(dim, []) else " "
        print(
            f"  {mapped}{dim:<26} {m['precision']:>7.3f} {m['recall']:>7.3f} "
            f"{m['f1']:>7.3f} {m['support_positive']:>7d} {m['predicted_positive']:>8d}",
            flush=True,
        )

    agg = metrics["_aggregate"]
    print("-" * len(header), flush=True)
    print(f"  Micro F1 (all dims):        {agg['micro_f1']:.4f}", flush=True)
    print(f"  Macro F1 (all dims):        {agg['macro_f1']:.4f}", flush=True)
    print(f"  Micro F1 (mapped dims):     {agg['mapped_micro_f1']:.4f}", flush=True)
    print(f"  Macro F1 (mapped dims):     {agg['mapped_macro_f1']:.4f}", flush=True)
    print(f"  N samples:                  {agg['n_samples']}", flush=True)

    # Per-benchmark breakdown
    print("\n" + "=" * 70, flush=True)
    print("PER-BENCHMARK BREAKDOWN", flush=True)
    print("=" * 70, flush=True)
    for bench in sorted(results_df["benchmark"].unique()):
        bench_df = results_df[results_df["benchmark"] == bench]
        yt = bench_df[[f"gt_{d}" for d in dim_names]].values
        yp = bench_df[[f"pred_{d}" for d in dim_names]].values
        bm = compute_metrics(yt, yp, dim_names)
        ba = bm["_aggregate"]
        print(f"\n  {bench} (n={len(bench_df)}):", flush=True)
        print(f"    Micro F1 (all):    {ba['micro_f1']:.4f}", flush=True)
        print(f"    Macro F1 (all):    {ba['macro_f1']:.4f}", flush=True)
        print(f"    Micro F1 (mapped): {ba['mapped_micro_f1']:.4f}", flush=True)
        for dim in dim_names:
            m = bm[dim]
            if m["support_positive"] > 0 or m["predicted_positive"] > 0:
                print(
                    f"      {dim:<26} P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f} sup={m['support_positive']} pred={m['predicted_positive']}",
                    flush=True,
                )

    # Raw rule stats
    print("\n" + "=" * 70, flush=True)
    print("WISIO RAW RULE DETECTION RATES", flush=True)
    print("=" * 70, flush=True)
    for rule in WISIO_RULES:
        n_det = results_df[f"wisio_{rule}"].sum()
        print(f"  {rule:<35} {n_det:>4}/{len(results_df)} ({100*n_det/len(results_df):.1f}%)", flush=True)

    # Save metrics JSON
    metrics_output = {
        "overall": agg,
        "per_dimension": {d: metrics[d] for d in dim_names},
        "per_benchmark": {},
        "errors_count": len(errors),
        "total_unique_jobs": len(unique_jobs),
        "successful_jobs": len(wisio_cache) - len(errors),
        "total_label_rows": len(results_df),
        "elapsed_seconds": elapsed_total,
    }
    for bench in sorted(results_df["benchmark"].unique()):
        bench_df = results_df[results_df["benchmark"] == bench]
        yt = bench_df[[f"gt_{d}" for d in dim_names]].values
        yp = bench_df[[f"pred_{d}" for d in dim_names]].values
        bm = compute_metrics(yt, yp, dim_names)
        metrics_output["per_benchmark"][bench] = {
            "aggregate": bm["_aggregate"],
            "per_dimension": {d: bm[d] for d in dim_names},
        }

    with open(OUTPUT_DIR / "wisio_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\nSaved metrics to {OUTPUT_DIR / 'wisio_metrics.json'}", flush=True)

    if errors:
        with open(OUTPUT_DIR / "wisio_errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors to {OUTPUT_DIR / 'wisio_errors.json'}", flush=True)


if __name__ == "__main__":
    main()
