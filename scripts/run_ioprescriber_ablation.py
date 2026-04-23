#!/usr/bin/env python3
"""
Ablation Study for IOPrescriber — IOSage paper Table 4.

Runs 6 ablation conditions to prove each component adds value:
  A1: Full system vs LLM alone (no ML detection, raw Darshan to LLM)
  A2: Full system vs No KB (LLM gets ML output but no RAG retrieval)
  A3: Full system vs No SHAP (LLM gets labels but not feature attribution)
  A4: Full system vs ML-only (detection without LLM recommendation)
  A5: Full system vs No ML (IOAgent-style: KB + LLM, no ML detection)
  A6: Comparison across models (already done in run_llm_evaluation.py)

Each ablation uses 4 representative workloads (one per bottleneck type)
with Claude Sonnet as the LLM.

Key metric: Groundedness score (does removing a component hurt recommendation quality?)
Secondary: recommendation relevance, number of recommendations, latency.

PerfCoder showed ML planner + LLM = 4.82x vs LLM alone = 1.96x.
ECO showed structured prompting = 7.81x vs conventional = 1.99x.
We must show similar gaps.

Usage:
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --ablation A1  # single ablation
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

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

DIM_DESCRIPTIONS = {
    "access_granularity": "Small I/O operations (<1MB transfer size)",
    "metadata_intensity": "Excessive metadata operations",
    "parallelism_efficiency": "Load imbalance across ranks",
    "access_pattern": "Random (non-sequential) access",
    "interface_choice": "Wrong I/O interface (POSIX instead of MPI-IO)",
    "file_strategy": "File-per-process explosion",
    "throughput_utilization": "Low throughput (excessive sync/single-OST)",
    "healthy": "No bottleneck detected",
}


def select_ablation_workloads(test_feat, test_labels, n=4):
    """Select 4 workloads covering different bottleneck types."""
    selected = []
    target_dims = ["access_granularity", "interface_choice", "metadata_intensity", "throughput_utilization"]

    for dim in target_dims:
        mask = test_labels[dim] == 1
        candidates = test_feat.index[mask.values].tolist()
        if candidates:
            selected.append(candidates[0])

    # Fill if we don't have enough
    while len(selected) < n:
        for idx in test_feat.index.tolist():
            if idx not in selected:
                selected.append(idx)
                break

    return selected[:n]


def run_full_system(pipeline, features, workload_name):
    """A0: Full system (baseline) — ML + SHAP + KB + LLM."""
    return pipeline.analyze(features, workload_name=f"FULL_{workload_name}")


def run_no_ml(recommender, retriever, features, workload_name):
    """A1: No ML detection — give LLM raw Darshan summary only (IOAgent-style)."""
    # No ML predictions, no SHAP — just raw features to LLM
    summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s",
                     "POSIX_WRITES", "POSIX_READS", "POSIX_FSYNCS"]
    darshan_summary = {k: round(float(features.get(k, 0)), 4)
                       for k in summary_keys if features.get(k, 0) != 0}

    system_prompt = """You are an HPC I/O performance expert. Analyze these Darshan counters
and recommend code-level I/O optimizations. Be specific about what to change.
Respond in JSON with: diagnosis, recommendations (each with code_before, code_after, expected_speedup)."""

    user_prompt = f"""Analyze this job's Darshan I/O profile:

{json.dumps(darshan_summary, indent=2)}

Provide specific code-level optimization recommendations in JSON format.
"""

    raw_response, metadata = recommender.call_llm(system_prompt, user_prompt)
    parsed, parse_error = recommender.parse_response(raw_response)

    return {
        "condition": "A1_no_ml",
        "workload": workload_name,
        "has_ml": False, "has_shap": False, "has_kb": False,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": {"groundedness_score": 0.0, "n_recommendations": 0, "n_grounded": 0,
                          "note": "No KB provided — groundedness N/A"},
        "parse_error": parse_error,
    }


def run_no_kb(pipeline, features, workload_name):
    """A2: No KB — ML + SHAP but LLM gets no benchmark evidence."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = np.array([[features.get(col, 0) for col in pipeline.detector.feature_cols]],
                  dtype=np.float32)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)

    summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
    darshan_summary = {k: round(float(features.get(k, 0)), 4)
                       for k in summary_keys if features.get(k, 0) != 0}

    # Build prompt WITHOUT KB entries
    detection_str = "\n".join(
        f"  - {dim}: {predictions[dim]:.2f} — {DIM_DESCRIPTIONS.get(dim, '')}"
        for dim in detected
    )
    shap_str = ""
    for dim, feats in shap_features.items():
        if feats:
            shap_str += f"\n  {dim}:\n"
            for f in feats[:5]:
                shap_str += f"    - {f['feature']} = {f['value']:.4f} (|SHAP|={f['abs_importance']:.4f})\n"

    system_prompt = """You are an HPC I/O performance expert. You are given ML detection results
and SHAP feature attributions. Recommend specific code-level I/O optimizations.
Respond in JSON with: diagnosis, recommendations (each with code_before, code_after, expected_speedup)."""

    user_prompt = f"""## Detected Bottlenecks (ML classifier):
{detection_str}

## Key Features (SHAP attribution):
{shap_str}

## Darshan Summary:
{json.dumps(darshan_summary, indent=2)}

Note: No benchmark evidence available. Base recommendations on general I/O best practices.
Respond in JSON format.
"""

    raw_response, metadata = pipeline.recommender.call_llm(system_prompt, user_prompt)
    parsed, parse_error = pipeline.recommender.parse_response(raw_response)

    return {
        "condition": "A2_no_kb",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": {"groundedness_score": 0.0, "note": "No KB — groundedness N/A"},
        "parse_error": parse_error,
    }


def run_no_shap(pipeline, features, workload_name):
    """A3: No SHAP — ML + KB but LLM gets bottleneck labels only, no feature attribution."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    kb_entries = pipeline.retriever.retrieve(detected, features)

    summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
    darshan_summary = {k: round(float(features.get(k, 0)), 4)
                       for k in summary_keys if features.get(k, 0) != 0}

    # Build prompt WITHOUT SHAP features
    empty_shap = {}
    parsed, groundedness, metadata, raw = pipeline.recommender.recommend(
        predictions, detected, empty_shap, kb_entries, darshan_summary
    )

    return {
        "condition": "A3_no_shap",
        "workload": workload_name,
        "has_ml": True, "has_shap": False, "has_kb": True,
        "detected": detected,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": groundedness,
    }


def run_ml_only(pipeline, features, workload_name):
    """A4: ML-only — detection + SHAP, no LLM recommendation."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = np.array([[features.get(col, 0) for col in pipeline.detector.feature_cols]],
                  dtype=np.float32)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)

    return {
        "condition": "A4_ml_only",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "predictions": predictions,
        "shap_top_features": {dim: feats[:3] for dim, feats in shap_features.items()},
        "recommendation": None,
        "metadata": {"latency_ms": 0, "tokens_input": 0, "tokens_output": 0},
        "groundedness": {"groundedness_score": None, "note": "No LLM — no recommendations to ground"},
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study for IOPrescriber")
    parser.add_argument("--ablation", default="all", choices=["all", "A1", "A2", "A3", "A4"])
    parser.add_argument("--n-workloads", type=int, default=4)
    args = parser.parse_args()

    # Load env
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    key, val = line.strip().split("=", 1)
                    os.environ[key.replace("export ", "").strip()] = val.strip().strip('"')

    from src.ioprescriber.pipeline import IOPrescriber

    pipeline = IOPrescriber(llm_model="claude-sonnet")

    # Load test data
    test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
    test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

    workload_indices = select_ablation_workloads(test_feat, test_labels, args.n_workloads)
    logger.info("Selected %d workloads for ablation", len(workload_indices))

    all_results = {}
    ablations_to_run = ["A0", "A1", "A2", "A3", "A4"] if args.ablation == "all" else ["A0", args.ablation]

    for abl in ablations_to_run:
        logger.info("")
        logger.info("=" * 60)
        logger.info("ABLATION %s", abl)
        logger.info("=" * 60)

        abl_results = []

        for w_idx, idx in enumerate(workload_indices):
            features = test_feat.iloc[idx].to_dict()
            label_row = test_labels.iloc[idx]
            name = f"{label_row.get('benchmark', '?')}_{label_row.get('scenario', '?')}"

            logger.info("  Workload %d/%d: %s", w_idx + 1, len(workload_indices), name)

            try:
                if abl == "A0":
                    result = run_full_system(pipeline, features, name)
                elif abl == "A1":
                    result = run_no_ml(pipeline.recommender, pipeline.retriever, features, name)
                elif abl == "A2":
                    result = run_no_kb(pipeline, features, name)
                elif abl == "A3":
                    result = run_no_shap(pipeline, features, name)
                elif abl == "A4":
                    result = run_ml_only(pipeline, features, name)
                else:
                    continue

                abl_results.append(result)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                abl_results.append({"condition": abl, "workload": name, "error": str(e)})

        all_results[abl] = abl_results

    # Summary Table
    logger.info("")
    logger.info("=" * 80)
    logger.info("TABLE 4: ABLATION STUDY RESULTS")
    logger.info("=" * 80)

    header = f"{'Condition':<25s} {'Components':<30s} {'Ground.':<10s} {'#Recs':<8s} {'Latency':<10s}"
    logger.info(header)
    logger.info("-" * 83)

    condition_names = {
        "A0": ("Full System", "ML+SHAP+KB+LLM"),
        "A1": ("No ML (IOAgent-style)", "KB+LLM only"),
        "A2": ("No KB", "ML+SHAP+LLM"),
        "A3": ("No SHAP", "ML+KB+LLM"),
        "A4": ("ML-Only (AIIO-style)", "ML+SHAP only"),
    }

    for abl, results in all_results.items():
        name, components = condition_names.get(abl, (abl, "?"))
        valid = [r for r in results if "error" not in r]

        gnd_scores = []
        n_recs_list = []
        latencies = []

        for r in valid:
            g = r.get("groundedness", {})
            if g.get("groundedness_score") is not None:
                gnd_scores.append(g["groundedness_score"])
            rec = r.get("recommendation")
            if rec and isinstance(rec, dict):
                n_recs_list.append(len(rec.get("recommendations", [])))
            elif r.get("step4_recommendation", {}).get("parsed"):
                n_recs_list.append(len(r["step4_recommendation"]["parsed"].get("recommendations", [])))
            m = r.get("metadata", {})
            if not m:
                m = r.get("step4_recommendation", {}).get("metadata", {})
            if m and m.get("latency_ms"):
                latencies.append(m["latency_ms"])

        gnd = f"{np.mean(gnd_scores):.3f}" if gnd_scores else "N/A"
        recs = f"{np.mean(n_recs_list):.1f}" if n_recs_list else "N/A"
        lat = f"{np.mean(latencies):.0f}ms" if latencies else "N/A"

        logger.info(f"{name:<25s} {components:<30s} {gnd:<10s} {recs:<8s} {lat:<10s}")

    logger.info("=" * 80)

    # Save
    results_dir = PROJECT_DIR / "results" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"ablation_results_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
