#!/usr/bin/env python3
"""
Fair comparison: Run IOSage with gpt-4.1-mini (same model as IOAgent baseline).

Uses the BOOST model and new test splits for apples-to-apples comparison.
LLM calls go through OpenRouter with gpt-4.1-mini model.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["gpt-4.1-mini", "gpt-4o", "claude-sonnet-4", "llama-3.1-70b-instruct"]

# OpenRouter model id mapping (added to rec_mod.MODELS at runtime)
OPENROUTER_MAP = {
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
}

OUTPUT_DIR = PROJECT_DIR / "results" / "boost_experiment" / "iosage_full488"

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


def select_all_traces(test_feat, test_labels, only_benchmark=None, n_max=None):
    """Process ALL test traces (no diverse subsampling).
    Optionally filter by benchmark or limit to first n_max for smoke test."""
    if only_benchmark is not None:
        mask = test_labels['benchmark'] == only_benchmark
        indices = test_feat.index[mask.values].tolist()
    else:
        indices = test_feat.index.tolist()
    if n_max is not None:
        indices = indices[:n_max]
    return indices


def select_diverse_workloads(test_feat, test_labels, n_workloads=12):
    """Select diverse workloads covering all bottleneck types."""
    selected = []
    seen_scenarios = set()

    # First pass: get at least 1 workload per bottleneck type
    for dim in DIMENSIONS:
        if dim == "healthy":
            continue
        mask = test_labels[dim] == 1
        candidates = test_feat.index[mask.values].tolist()
        for idx in candidates:
            scenario = test_labels.iloc[idx].get("scenario", "")
            base_scenario = "_".join(scenario.split("_")[:3])  # deduplicate reps
            if base_scenario not in seen_scenarios and len(selected) < n_workloads:
                selected.append(idx)
                seen_scenarios.add(base_scenario)
                break

    # Second pass: add healthy workload
    healthy_mask = test_labels["healthy"] == 1
    for idx in test_feat.index[healthy_mask.values].tolist():
        scenario = test_labels.iloc[idx].get("scenario", "")
        base_scenario = "_".join(scenario.split("_")[:3])
        if base_scenario not in seen_scenarios and len(selected) < n_workloads:
            selected.append(idx)
            seen_scenarios.add(base_scenario)
            break

    # Fill remaining slots with diverse workloads
    for idx in test_feat.index.tolist():
        if idx not in selected and len(selected) < n_workloads:
            scenario = test_labels.iloc[idx].get("scenario", "")
            base_scenario = "_".join(scenario.split("_")[:3])
            if base_scenario not in seen_scenarios:
                selected.append(idx)
                seen_scenarios.add(base_scenario)

    return selected[:n_workloads]


def run_evaluation(models_to_test, n_runs, only_benchmark=None, n_max=None,
                   per_model_output_dir=None):
    """Run full evaluation across models on all 488 test traces.
    Supports per-model output dir for resume across SLURM jobs."""
    from src.ioprescriber.pipeline import IOPrescriber
    from src.ioprescriber import detector as det_mod
    from src.ioprescriber import recommender as rec_mod

    # Override detector to use BOOST model
    _orig_det_init = det_mod.Detector.__init__
    def _patched_det_init(self, model_path=None, config_path=None, threshold=0.3):
        boost_model = PROJECT_DIR / "results" / "boost_experiment" / "new_models" / "xgboost_biquality_w100_seed42.pkl"
        _orig_det_init(self, model_path=str(boost_model), config_path=config_path, threshold=threshold)
    det_mod.Detector.__init__ = _patched_det_init

    # Add all 4 LLMs to recommender model map (routed via OpenRouter)
    for short, full in OPENROUTER_MAP.items():
        rec_mod.MODELS[short] = full

    # Load test data
    test_feat = pd.read_parquet(PROJECT_DIR / "results" / "boost_experiment" / "new_splits" / "test_features.parquet")
    test_labels = pd.read_parquet(PROJECT_DIR / "results" / "boost_experiment" / "new_splits" / "test_labels.parquet")

    # Select all test traces (no diverse subsampling — process the full 488)
    workload_indices = select_all_traces(test_feat, test_labels,
                                         only_benchmark=only_benchmark, n_max=n_max)
    logger.info("Selected %d traces (benchmark=%s, n_max=%s)",
                len(workload_indices), only_benchmark, n_max)

    all_results = {}

    for model in models_to_test:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL: %s", model)
        logger.info("=" * 70)

        # Resume support: load existing results for this model if any
        model_results = []
        done_indices = set()
        if per_model_output_dir is not None:
            raw_path = per_model_output_dir / f"raw_results_{model}.json"
            if raw_path.exists():
                try:
                    with open(raw_path) as f:
                        model_results = json.load(f)
                    done_indices = {(r.get("workload_index"), r.get("run", 0))
                                    for r in model_results if "error" not in r}
                    logger.info("Resuming: loaded %d existing results, skipping done", len(model_results))
                except Exception as e:
                    logger.warning("Could not load existing results: %s", e)
                    model_results = []
                    done_indices = set()

        # Initialize pipeline for this model
        try:
            pipeline = IOPrescriber(llm_model=model)
        except Exception as e:
            logger.error("Failed to initialize %s: %s", model, e)
            continue

        for w_idx, idx in enumerate(workload_indices):
            features = test_feat.iloc[idx].to_dict()
            label_row = test_labels.iloc[idx]
            workload_name = f"{label_row.get('benchmark', '?')}_{label_row.get('scenario', '?')}"
            gt_labels = {d: int(label_row.get(d, 0)) for d in DIMENSIONS}

            for run in range(n_runs):
                if (int(idx), run) in done_indices:
                    continue
                logger.info("Trace %d/%d run %d: %s",
                            w_idx + 1, len(workload_indices), run + 1, workload_name)

                try:
                    result = pipeline.analyze(features, workload_name=f"{workload_name}_run{run}")
                    result["ground_truth_labels"] = gt_labels
                    result["workload_index"] = int(idx)
                    result["benchmark"] = str(label_row.get("benchmark", "?"))
                    result["run"] = run
                    result["model"] = model
                    model_results.append(result)
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    model_results.append({
                        "workload": workload_name,
                        "workload_index": int(idx),
                        "model": model,
                        "run": run,
                        "error": str(e),
                    })

                # Save incrementally for resume
                if per_model_output_dir is not None:
                    raw_path = per_model_output_dir / f"raw_results_{model}.json"
                    with open(raw_path, "w") as f:
                        json.dump(model_results, f, indent=2, default=str)

        all_results[model] = model_results

    return all_results, workload_indices


def compute_summary(all_results):
    """Compute summary statistics for paper Tables V and VI.
    Adds Recommendation Precision (Rec.P) — fraction of recs that target an
    actual ground-truth bottleneck dimension."""
    summary = {}

    for model, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if not valid:
            continue

        groundedness_scores = []
        latencies = []
        token_counts = []
        n_recommendations = []
        parse_errors = 0
        rec_precision_scores = []  # Per-workload Rec.P
        n_traces_with_recs = 0
        n_healthy = 0

        for r in valid:
            # Note: actual key is step3_recommendation (verified from raw output)
            rec = r.get("step3_recommendation") or r.get("step4_recommendation") or {}
            g = rec.get("groundedness", {}) if isinstance(rec, dict) else {}
            m = rec.get("metadata", {}) if isinstance(rec, dict) else {}
            gt = r.get("ground_truth_labels", {})

            if g and g.get("groundedness_score") is not None:
                groundedness_scores.append(g["groundedness_score"])
            if m and m.get("latency_ms"):
                latencies.append(m["latency_ms"])
            if m:
                tokens = m.get("tokens_input", 0) + m.get("tokens_output", 0)
                if tokens > 0:
                    token_counts.append(tokens)
            parsed = rec.get("parsed") if isinstance(rec, dict) else None
            if parsed:
                recs = parsed.get("recommendations", [])
                n_recommendations.append(len(recs))
                if len(recs) > 0:
                    n_traces_with_recs += 1
                    # Rec.P: fraction of recs whose bottleneck_dimension is in GT positives
                    gt_positive_dims = {d for d in DIMENSIONS if gt.get(d, 0) == 1 and d != "healthy"}
                    n_correct = sum(1 for rr in recs
                                    if rr.get("bottleneck_dimension") in gt_positive_dims)
                    rec_precision_scores.append(n_correct / len(recs))
            elif rec:
                # rec exists but no parsed structure means parse failure
                parse_errors += 1

            # Count healthy traces
            if gt.get("healthy", 0) == 1:
                n_healthy += 1

        summary[model] = {
            "n_valid": len(valid),
            "n_errors": len(results) - len(valid),
            "n_parse_errors": parse_errors,
            "n_healthy": n_healthy,
            "n_traces_with_recs": n_traces_with_recs,
            "groundedness_mean": float(np.mean(groundedness_scores)) if groundedness_scores else 0,
            "groundedness_std": float(np.std(groundedness_scores)) if groundedness_scores else 0,
            "rec_precision_mean": float(np.mean(rec_precision_scores)) if rec_precision_scores else 0,
            "rec_precision_std": float(np.std(rec_precision_scores)) if rec_precision_scores else 0,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else 0,
            "latency_std_ms": float(np.std(latencies)) if latencies else 0,
            "tokens_mean": float(np.mean(token_counts)) if token_counts else 0,
            "avg_recommendations": float(np.mean(n_recommendations)) if n_recommendations else 0,
        }

    return summary


def print_table3(summary):
    """Print Table 3: LLM Recommendation Quality."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TABLE 3: LLM Recommendation Quality")
    logger.info("=" * 80)
    header = f"{'Model':<20s} {'Ground.':<12s} {'Latency(ms)':<14s} {'Tokens':<10s} {'#Recs':<8s} {'Parse%':<8s}"
    logger.info(header)
    logger.info("-" * 72)

    for model, s in summary.items():
        gnd = f"{s['groundedness_mean']:.3f}±{s['groundedness_std']:.3f}"
        lat = f"{s['latency_mean_ms']:.0f}±{s['latency_std_ms']:.0f}"
        tok = f"{s['tokens_mean']:.0f}"
        recs = f"{s['avg_recommendations']:.1f}"
        parse = f"{100*(1-s['n_parse_errors']/max(s['n_valid'],1)):.0f}%"
        logger.info(f"{model:<20s} {gnd:<12s} {lat:<14s} {tok:<10s} {recs:<8s} {parse:<8s}")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run IOSage full pipeline on 488 test traces")
    parser.add_argument("--model", default="gpt-4.1-mini", choices=["all"] + MODELS,
                        help="LLM to use (one per SLURM job)")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Runs per trace (1 by default; 5 for multi-seed)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Process only this benchmark (optional)")
    parser.add_argument("--n-max", type=int, default=None,
                        help="Process only first N traces (for smoke tests)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    # Load env
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    key, val = line.strip().split("=", 1)
                    key = key.replace("export ", "").strip()
                    val = val.strip().strip('"')
                    os.environ[key] = val

    models = MODELS if args.model == "all" else [args.model]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("IOSage full pipeline: %d model(s) %s, n_runs=%d, benchmark=%s, n_max=%s",
                len(models), models, args.n_runs, args.benchmark, args.n_max)

    all_results, workload_indices = run_evaluation(
        models, args.n_runs,
        only_benchmark=args.benchmark,
        n_max=args.n_max,
        per_model_output_dir=output_dir,
    )

    # Compute summary
    summary = compute_summary(all_results)
    print_table3(summary)

    # Save final summary (per-model raw results already saved incrementally during run)
    suffix = f"_{args.model}" if args.model != "all" else ""
    if args.benchmark:
        suffix += f"_{args.benchmark}"
    if args.n_max:
        suffix += f"_smoke{args.n_max}"
    summary_path = output_dir / f"evaluation_summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("Per-model raw results in: %s", output_dir)
    logger.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
