#!/usr/bin/env python3
"""
Run systematic LLM evaluation for SC 2026 paper.

Evaluates IOPrescriber Track B (single-shot recommendation) across:
  - 12 diverse workloads (covering all 8 bottleneck types)
  - 3 LLM models (Claude Sonnet, GPT-4o, Llama-70b)
  - 5 runs per combination (for variance at temperature=0)

Produces:
  - Table 3: LLM Recommendation Quality (multi-model comparison)
  - Groundedness scores per model
  - Latency and token cost analysis

Usage:
    python scripts/run_llm_evaluation.py                    # All models, 12 workloads
    python scripts/run_llm_evaluation.py --model claude-sonnet --n-workloads 5
    python scripts/run_llm_evaluation.py --n-runs 1         # Quick test (1 run)
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

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["claude-sonnet", "gpt-4o", "llama-70b"]

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


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


def run_evaluation(models_to_test, n_workloads, n_runs):
    """Run full evaluation across models, workloads, and runs."""
    from src.ioprescriber.pipeline import IOPrescriber

    # Load test data
    test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
    test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

    # Select diverse workloads
    workload_indices = select_diverse_workloads(test_feat, test_labels, n_workloads)
    logger.info("Selected %d diverse workloads", len(workload_indices))

    for idx in workload_indices:
        label_row = test_labels.iloc[idx]
        active = [d for d in DIMENSIONS if label_row.get(d, 0) == 1]
        logger.info("  %s/%s: %s", label_row.get("benchmark", "?"),
                    label_row.get("scenario", "?"), active)

    all_results = {}

    for model in models_to_test:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL: %s", model)
        logger.info("=" * 70)

        # Initialize pipeline for this model
        try:
            pipeline = IOPrescriber(llm_model=model)
        except Exception as e:
            logger.error("Failed to initialize %s: %s", model, e)
            continue

        model_results = []

        for w_idx, idx in enumerate(workload_indices):
            features = test_feat.iloc[idx].to_dict()
            label_row = test_labels.iloc[idx]
            workload_name = f"{label_row.get('benchmark', '?')}_{label_row.get('scenario', '?')}"
            gt_labels = {d: int(label_row.get(d, 0)) for d in DIMENSIONS}

            for run in range(n_runs):
                logger.info("")
                logger.info("Workload %d/%d, Run %d/%d: %s",
                            w_idx + 1, len(workload_indices), run + 1, n_runs, workload_name)

                try:
                    result = pipeline.analyze(features, workload_name=f"{workload_name}_run{run}")
                    result["ground_truth_labels"] = gt_labels
                    result["workload_index"] = int(idx)
                    result["run"] = run
                    result["model"] = model
                    model_results.append(result)
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    model_results.append({
                        "workload": workload_name,
                        "model": model,
                        "run": run,
                        "error": str(e),
                    })

        all_results[model] = model_results

    return all_results, workload_indices


def compute_summary(all_results):
    """Compute summary statistics for paper Table 3."""
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

        for r in valid:
            rec = r.get("step4_recommendation", {})
            g = rec.get("groundedness", {})
            m = rec.get("metadata", {})

            if g and g.get("groundedness_score") is not None:
                groundedness_scores.append(g["groundedness_score"])
            if m and m.get("latency_ms"):
                latencies.append(m["latency_ms"])
            if m:
                tokens = m.get("tokens_input", 0) + m.get("tokens_output", 0)
                if tokens > 0:
                    token_counts.append(tokens)
            if rec.get("parsed"):
                n_recommendations.append(len(rec["parsed"].get("recommendations", [])))
            else:
                parse_errors += 1

        summary[model] = {
            "n_valid": len(valid),
            "n_errors": len(results) - len(valid),
            "n_parse_errors": parse_errors,
            "groundedness_mean": np.mean(groundedness_scores) if groundedness_scores else 0,
            "groundedness_std": np.std(groundedness_scores) if groundedness_scores else 0,
            "latency_mean_ms": np.mean(latencies) if latencies else 0,
            "latency_std_ms": np.std(latencies) if latencies else 0,
            "tokens_mean": np.mean(token_counts) if token_counts else 0,
            "avg_recommendations": np.mean(n_recommendations) if n_recommendations else 0,
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
    parser = argparse.ArgumentParser(description="Run LLM evaluation for SC 2026")
    parser.add_argument("--model", default="all", choices=["all"] + MODELS)
    parser.add_argument("--n-workloads", type=int, default=12)
    parser.add_argument("--n-runs", type=int, default=1, help="Runs per workload (5 for paper)")
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

    logger.info("LLM Evaluation: %d models × %d workloads × %d runs = %d total calls",
                len(models), args.n_workloads, args.n_runs,
                len(models) * args.n_workloads * args.n_runs)

    all_results, workload_indices = run_evaluation(models, args.n_workloads, args.n_runs)

    # Compute summary
    summary = compute_summary(all_results)
    print_table3(summary)

    # Save everything
    results_dir = PROJECT_DIR / "results" / "llm_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    results_path = results_dir / f"evaluation_results_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("Results saved: %s", results_path)
    logger.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
