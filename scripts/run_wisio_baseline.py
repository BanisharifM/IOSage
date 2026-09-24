"""Run WisIO on benchmark traces with one prediction per label row.

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

"""

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
os.environ["DASK_DISTRIBUTED__LOGGING__DISTRIBUTED"] = "error"
os.environ["DASK_LOGGING__DISTRIBUTED"] = "error"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def assign_trace_paths(labels_df, benchmark_logs_dir):
    """Assign every label row to one trace, enforcing group cardinality."""
    required = {"job_id", "benchmark", "scenario", "n_darshan_files"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels table lacks columns: {sorted(missing)}")

    log_root = Path(benchmark_logs_dir)
    files_by_group = {}
    for benchmark in labels_df["benchmark"].astype(str).unique():
        bench_dir = log_root / benchmark
        if not bench_dir.is_dir():
            raise FileNotFoundError(f"benchmark log directory does not exist: {bench_dir}")
        for path in bench_dir.glob("*.darshan"):
            match = re.search(r"_id([^-]+)-", path.name)
            if match:
                files_by_group.setdefault((benchmark, match.group(1)), []).append(path)

    assignments = {}
    for (benchmark, job_id), group in labels_df.groupby(
            ["benchmark", "job_id"], sort=False):
        matches = sorted(files_by_group.get((str(benchmark), str(job_id)), []))
        expected = group["n_darshan_files"].astype(int)
        if (expected < 1).any():
            raise ValueError(f"{benchmark}/{job_id}: invalid trace count")
        if len(matches) != int(expected.sum()):
            raise ValueError(
                f"{benchmark}/{job_id}: labels require {int(expected.sum())} "
                f"traces but {len(matches)} match")
        offset = 0
        for row_index, count in zip(group.index, expected):
            assignments[row_index] = tuple(
                path.resolve() for path in matches[offset:offset + count])
            offset += count

    if len(assignments) != len(labels_df):
        raise ValueError(
            f"assigned {len(assignments)} traces for {len(labels_df)} rows")
    assigned_paths = [path for paths in assignments.values() for path in paths]
    if len(set(assigned_paths)) != len(assigned_paths):
        raise ValueError("a Darshan trace was assigned to more than one label row")
    return assignments


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

    results["_aggregate"] = {
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_f1": f1_macro,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", default=str(
            PROJECT_ROOT / "data/processed/resubmission/benchmark/labels.parquet"))
    parser.add_argument(
        "--benchmark-logs", default=str(PROJECT_ROOT / "data/benchmark_logs"))
    parser.add_argument(
        "--runtime-python",
        help="optional package directory; omit after installing requirements-wisio.txt")
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "results/wisio_baseline"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("wisio_baseline")
    labels_path = Path(args.labels).resolve()
    logs_dir = Path(args.benchmark_logs).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path, description in (
            (labels_path, "labels table"),
            (logs_dir, "benchmark log directory")):
        if not path.exists():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    if args.runtime_python:
        runtime_python = Path(args.runtime_python).resolve()
        if not runtime_python.is_dir():
            raise FileNotFoundError(f"WisIO Python package directory does not exist: {runtime_python}")
        sys.path.insert(0, str(runtime_python))

    # Load ground-truth labels
    labels_df = pd.read_parquet(labels_path)
    assignments = assign_trace_paths(labels_df, logs_dir)
    assigned_count = sum(len(paths) for paths in assignments.values())
    logger.info("Assigned %d label rows to %d distinct traces",
                len(labels_df), assigned_count)

    # Initialize Dask
    try:
        from dask.distributed import LocalCluster, Client
        from wisio.darshan import DarshanAnalyzer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WisIO dependencies are missing; install requirements-wisio.txt "
            "or pass --runtime-python"
        ) from exc

    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, memory_limit="4GB", silence_logs=50
    )
    client = Client(cluster)
    logger.info("Dask client started")

    trace_results = {}
    errors = []
    t_start = time.time()
    items = list(assignments.items())

    for position, (row_index, trace_paths) in enumerate(items, start=1):
        try:
            rule_flags = {rule: False for rule in WISIO_RULES}
            for trace_path in trace_paths:
                analyzer = DarshanAnalyzer(
                    checkpoint=False, checkpoint_dir="",
                    bottleneck_dir="/tmp/wisio_baseline_bot", verbose=False,
                )
                result = analyzer.analyze_trace(
                    trace_path=str(trace_path),
                    percentile=0.9,
                    view_types=["file_name", "proc_name"],
                    metrics=["iops"],
                    exclude_bottlenecks=[],
                    exclude_characteristics=[],
                )
                if result._bottlenecks is not None:
                    bot_df = result._bottlenecks.compute()
                    for rule in WISIO_RULES:
                        detected = rule in bot_df.columns and bool(bot_df[rule].any())
                        rule_flags[rule] = rule_flags[rule] or detected

            trace_results[row_index] = rule_flags

        except Exception as exc:
            errors.append({
                "row_index": int(row_index),
                "job_id": int(labels_df.loc[row_index, "job_id"]),
                "traces": [str(path) for path in trace_paths],
                "error": str(exc),
            })
            logger.error("WisIO failed for row %s: %s", row_index, exc)

        if position % 10 == 0:
            elapsed = time.time() - t_start
            rate = position / elapsed
            eta = (len(items) - position) / rate if rate > 0 else 0
            logger.info("%d/%d, %.2f traces/s, %.0fs remaining, %d errors",
                        position, len(items), rate, eta, len(errors))

    elapsed_total = time.time() - t_start
    client.close()
    cluster.close()

    if errors:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / "wisio_errors.json"
        error_path.write_text(json.dumps(errors, indent=2))
        raise RuntimeError(
            f"WisIO failed on {len(errors)} of {len(items)} traces; "
            f"details: {error_path}")
    if len(trace_results) != len(labels_df):
        raise RuntimeError(
            f"WisIO returned {len(trace_results)} results for {len(labels_df)} rows")

    all_results = []
    for row_index, row in labels_df.iterrows():
        rule_flags = trace_results[row_index]
        taxonomy_preds = wisio_rules_to_taxonomy(rule_flags)

        record = {
            "sample_id": f"{row['benchmark']}/{row['scenario']}/{row_index}",
            "traces": [str(path) for path in assignments[row_index]],
            "job_id": row["job_id"],
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

    results_df = pd.DataFrame(all_results)
    if len(results_df) != len(labels_df) or not results_df["sample_id"].is_unique:
        raise RuntimeError("prediction output is not one-to-one with label rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_dir / "wisio_predictions.parquet", index=False)
    results_df.to_csv(output_dir / "wisio_predictions.csv", index=False)

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
        "total_traces": len(items),
        "successful_traces": len(trace_results),
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

    with open(output_dir / "wisio_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info("Saved complete WisIO evaluation to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
