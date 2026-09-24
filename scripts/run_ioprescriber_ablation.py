#!/usr/bin/env python3
"""
Ablation Study for IOPrescriber - IOSage paper Table 4.

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
from pathlib import Path

import numpy as np
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.models.biquality import load_final_benchmark_test_frames  # noqa: E402
from src.data.label_rules import DIMENSION_NAMES  # noqa: E402

DIMENSIONS = list(DIMENSION_NAMES)

DIM_DESCRIPTIONS = {
    "access_granularity": "Many requests no larger than 1 MiB",
    "metadata_intensity": "Metadata calls dominate recorded I/O time",
    "parallelism_efficiency": "Load imbalance across ranks",
    "access_pattern": "Nonsequential POSIX access",
    "request_alignment": "File-offset misalignment",
    "interface_choice": "Independent MPI-IO without collective calls",
    "file_strategy": "Many small data files",
    "throughput_utilization": "A synchronous durability call after each write",
    "healthy": "All registered patterns are absent and observable",
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
    """A0: Full system (baseline) - ML + SHAP + KB + LLM."""
    result = pipeline.analyze(features, workload_name=f"FULL_{workload_name}")
    from src.ioprescriber.contracts import validate_pipeline_result
    validate_pipeline_result(result)
    section = result["recommendation"]
    return {
        "condition": "A0_full", "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": True,
        "detected": result["detection"]["detected"],
        "recommendation": section["parsed"],
        "metadata": section["metadata"] or {},
        "groundedness": section["groundedness"] or {},
    }


def run_no_ml(recommender, retriever, features, workload_name):
    """A1: No ML detection - give LLM raw Darshan summary only (IOAgent-style)."""
    # No ML predictions, no SHAP - just raw features to LLM
    summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s",
                     "POSIX_WRITES", "POSIX_READS", "POSIX_FSYNCS"]
    darshan_summary = {k: round(float(features.get(k, 0)), 4)
                       for k in summary_keys if features.get(k, 0) != 0}

    parsed, metadata, raw_response = recommender.recommend_without_evidence(
        [dimension for dimension in DIMENSIONS if dimension != "healthy"],
        darshan_summary)
    parse_error = None

    return {
        "condition": "A1_no_ml",
        "workload": workload_name,
        "has_ml": False, "has_shap": False, "has_kb": False,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": {"groundedness_score": 0.0, "n_recommendations": 0, "n_grounded": 0,
                          "note": "No KB provided - groundedness N/A"},
        "parse_error": parse_error,
    }


def run_no_kb(pipeline, features, workload_name):
    """A2: No KB - ML + SHAP but LLM gets no benchmark evidence."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = pipeline.detector.feature_vector(features)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)

    summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
    darshan_summary = {k: round(float(features.get(k, 0)), 4)
                       for k in summary_keys if features.get(k, 0) != 0}

    parsed, metadata, raw_response = pipeline.recommender.recommend_without_evidence(
        detected, darshan_summary, shap_features=shap_features)
    parse_error = None

    return {
        "condition": "A2_no_kb",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": {"groundedness_score": 0.0, "note": "No KB - groundedness N/A"},
        "parse_error": parse_error,
    }


def run_no_shap(pipeline, features, workload_name):
    """A3: No SHAP - ML + KB but LLM gets bottleneck labels only, no feature attribution."""
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
    """A4: ML-only - detection + SHAP, no LLM recommendation."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = pipeline.detector.feature_vector(features)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)

    return {
        "condition": "A4_ml_only",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "predictions": predictions,
        "shap_top_features": {dim: feats[:3] for dim, feats in shap_features.items()},
        "recommendation": None,
        "metadata": {"api_latency_ms": 0, "tokens_input": 0, "tokens_output": 0,
                     "cache_hit": False},
        "groundedness": {"groundedness_score": None, "note": "No LLM - no recommendations to ground"},
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study for IOPrescriber")
    parser.add_argument("--ablation", default="all", choices=["all", "A1", "A2", "A3", "A4"])
    parser.add_argument("--n-workloads", type=int, default=4)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--output-dir", required=True)
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

    pipeline = IOPrescriber(
        model_path=args.model_bundle, kb_path=args.knowledge_base,
        llm_model="claude-sonnet", use_shap=True)

    _, test_feat, test_labels, test_sample_ids = load_final_benchmark_test_frames(
        args.model_bundle)

    workload_indices = select_ablation_workloads(test_feat, test_labels, args.n_workloads)
    if len(workload_indices) != args.n_workloads:
        raise ValueError(
            f"requested {args.n_workloads} workloads but selected {len(workload_indices)}")
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
                raise AssertionError(f"unsupported ablation {abl}")
            abl_results.append(result)

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
            rec = r.get("recommendation")
            m = r.get("metadata", {})
            if g.get("groundedness_score") is not None:
                gnd_scores.append(g["groundedness_score"])
            if rec and isinstance(rec, dict):
                n_recs_list.append(len(rec.get("recommendations", [])))
            if m and m.get("api_latency_ms"):
                latencies.append(m["api_latency_ms"])

        gnd = f"{np.mean(gnd_scores):.3f}" if gnd_scores else "N/A"
        recs = f"{np.mean(n_recs_list):.1f}" if n_recs_list else "N/A"
        lat = f"{np.mean(latencies):.0f}ms" if latencies else "N/A"

        logger.info(f"{name:<25s} {components:<30s} {gnd:<10s} {recs:<8s} {lat:<10s}")

    logger.info("=" * 80)

    # Save
    results_dir = Path(args.output_dir).resolve()
    if results_dir.exists():
        raise FileExistsError(f"ablation output directory already exists: {results_dir}")
    results_dir.mkdir(parents=True)
    results_path = results_dir / "ablation_results.json"
    with results_path.open("x") as f:
        json.dump(all_results, f, indent=2, default=str)
    import hashlib
    with (results_dir / "ablation_manifest.json").open("x") as handle:
        json.dump({
            "schema_version": 1,
            "model_bundle_sha256": hashlib.sha256(Path(args.model_bundle).read_bytes()).hexdigest(),
            "knowledge_base_sha256": hashlib.sha256(Path(args.knowledge_base).read_bytes()).hexdigest(),
            "sample_ids": [test_sample_ids[index] for index in workload_indices],
            "conditions": ablations_to_run,
            "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        }, handle, indent=2)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
