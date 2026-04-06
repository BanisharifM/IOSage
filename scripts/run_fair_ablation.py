#!/usr/bin/env python3
"""
Fair Ablation Study for IOSage — SC 2026 Paper.

The original ablation had a flaw: "w/o ML classifier" also silently
disabled KB retrieval because the RAG query depends on ML-detected
dimensions. When ML returns ["unknown"], no KB entry matches.

This script runs a FAIR ablation with these conditions:
  C0: Full pipeline (ML + SHAP + KB + LLM)
  C1: w/o ML classifier (all 8 dims flagged, KB retrieves for all)
  C2: w/o knowledge grounding (ML + SHAP, no KB entries to LLM)
  C3: w/o feature attribution (ML + KB, no SHAP values to LLM)
  C4: Detection only (ML + SHAP, no LLM)
  C5: LLM only (no ML, no KB, no SHAP — raw Darshan to LLM)

Key difference from original: C1 flags ALL dimensions and retrieves
KB entries for all of them, isolating ML's contribution (focus/precision)
from KB's contribution (groundedness).

Results saved to results/ablation_fair/ (separate from original).

Usage:
    python scripts/run_fair_ablation.py
    python scripts/run_fair_ablation.py --n-workloads 8
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

BOTTLENECK_DIMS = [d for d in DIMENSIONS if d != "healthy"]

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


def select_ablation_workloads(test_feat, test_labels, n=8):
    """Select workloads covering diverse bottleneck types."""
    selected = []
    # One per bottleneck dimension
    for dim in BOTTLENECK_DIMS:
        if dim in test_labels.columns:
            mask = test_labels[dim] == 1
            candidates = test_feat.index[mask.values].tolist()
            if candidates:
                idx = candidates[0]
                if idx not in selected:
                    selected.append(idx)

    # Fill remaining slots with healthy samples
    if len(selected) < n:
        healthy_mask = test_labels.get("healthy", pd.Series(dtype=int)) == 1
        if healthy_mask.any():
            for idx in test_feat.index[healthy_mask.values].tolist():
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= n:
                        break

    return selected[:n]


def run_c0_full(pipeline, features, workload_name):
    """C0: Full pipeline — ML + SHAP + KB + LLM."""
    result = pipeline.analyze(features, workload_name=f"C0_{workload_name}")
    return _normalize_result(result, "C0_full", workload_name,
                             has_ml=True, has_shap=True, has_kb=True)


def run_c1_no_ml(pipeline, features, workload_name):
    """C1: No ML — flag ALL dimensions, retrieve KB for all, run LLM.

    This is the FAIR version: ML is removed but KB retrieval still works
    because we query with all bottleneck dimensions instead of ["unknown"].
    """
    # Fake predictions: all dimensions at confidence 1.0
    fake_predictions = {dim: 1.0 for dim in BOTTLENECK_DIMS}
    detected = list(BOTTLENECK_DIMS)  # All 7 bottleneck dims flagged

    # Retrieve KB entries for ALL dimensions
    kb_entries = pipeline.retriever.retrieve(detected, features)

    # No SHAP (depends on ML model)
    empty_shap = {}

    # Build summary
    darshan_summary = _build_darshan_summary(features)

    # Call LLM with all dims flagged + KB entries
    parsed, groundedness, metadata, raw = pipeline.recommender.recommend(
        fake_predictions, detected, empty_shap, kb_entries, darshan_summary
    )

    return {
        "condition": "C1_no_ml_with_kb",
        "workload": workload_name,
        "has_ml": False, "has_shap": False, "has_kb": True,
        "detected": detected,
        "n_kb_entries": len(kb_entries),
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": groundedness,
    }


def run_c2_no_kb(pipeline, features, workload_name):
    """C2: No KB — ML + SHAP detect, but LLM gets no KB entries."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = np.array([[features.get(col, 0) for col in pipeline.detector.feature_cols]],
                  dtype=np.float32)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)
    darshan_summary = _build_darshan_summary(features)

    # Build prompt WITHOUT KB entries — pass empty list
    parsed, groundedness, metadata, raw = pipeline.recommender.recommend(
        predictions, detected, shap_features, [], darshan_summary
    )

    return {
        "condition": "C2_no_kb",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": groundedness,
    }


def run_c3_no_shap(pipeline, features, workload_name):
    """C3: No SHAP — ML + KB but no feature attribution to LLM."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    kb_entries = pipeline.retriever.retrieve(detected, features)
    darshan_summary = _build_darshan_summary(features)

    # Empty SHAP
    empty_shap = {}

    parsed, groundedness, metadata, raw = pipeline.recommender.recommend(
        predictions, detected, empty_shap, kb_entries, darshan_summary
    )

    return {
        "condition": "C3_no_shap",
        "workload": workload_name,
        "has_ml": True, "has_shap": False, "has_kb": True,
        "detected": detected,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": groundedness,
    }


def run_c4_ml_only(pipeline, features, workload_name):
    """C4: ML detection only — no LLM recommendation."""
    predictions, detected = pipeline.detector.detect_from_features(features)
    X = np.array([[features.get(col, 0) for col in pipeline.detector.feature_cols]],
                  dtype=np.float32)
    shap_features = pipeline.explainer.explain(X, detected_dims=detected)

    return {
        "condition": "C4_detection_only",
        "workload": workload_name,
        "has_ml": True, "has_shap": True, "has_kb": False,
        "detected": detected,
        "predictions": {k: round(v, 4) for k, v in predictions.items()},
        "shap_top_features": {dim: feats[:3] for dim, feats in shap_features.items()},
        "recommendation": None,
        "metadata": {"latency_ms": 0, "tokens_input": 0, "tokens_output": 0},
        "groundedness": {"groundedness_score": None, "note": "No LLM — no recommendations"},
    }


def run_c5_llm_only(pipeline, features, workload_name):
    """C5: LLM only — no ML, no KB, no SHAP. Raw Darshan to LLM."""
    darshan_summary = _build_darshan_summary(features)

    system_prompt = """You are an HPC I/O performance expert. Analyze these Darshan counters
and recommend I/O optimizations. Be specific about what to change.

RULES:
1. You must cite a specific benchmark entry ID for each recommendation.
   If you cannot cite one, explicitly state that the recommendation is ungrounded.
2. Respond in JSON with: diagnosis, recommendations (each with fix, expected_speedup, kb_entry_id).
"""

    user_prompt = f"""Analyze this job's Darshan I/O profile and provide optimization recommendations:

{json.dumps(darshan_summary, indent=2)}

Respond in JSON format with diagnosis and recommendations.
Each recommendation MUST include a kb_entry_id field (or "none" if ungrounded).
"""

    raw_response, metadata = pipeline.recommender.call_llm(system_prompt, user_prompt)
    parsed, parse_error = pipeline.recommender.parse_response(raw_response)

    # Check groundedness manually
    n_recs = 0
    n_grounded = 0
    if parsed and isinstance(parsed, dict):
        recs = parsed.get("recommendations", [])
        n_recs = len(recs)
        for rec in recs:
            kb_id = rec.get("kb_entry_id", "none")
            if kb_id and kb_id != "none" and kb_id != "N/A":
                # Verify it exists in KB
                try:
                    kb = pipeline.retriever.kb
                    if any(e.get("entry_id") == kb_id for e in kb):
                        n_grounded += 1
                except Exception:
                    pass

    gnd_score = n_grounded / n_recs if n_recs > 0 else 0.0

    return {
        "condition": "C5_llm_only",
        "workload": workload_name,
        "has_ml": False, "has_shap": False, "has_kb": False,
        "recommendation": parsed,
        "metadata": metadata,
        "groundedness": {
            "groundedness_score": gnd_score,
            "n_recommendations": n_recs,
            "n_grounded": n_grounded,
        },
        "parse_error": parse_error,
    }


def _build_darshan_summary(features):
    """Extract key Darshan counters for LLM prompt."""
    summary_keys = [
        "nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ",
        "avg_write_size", "avg_read_size", "small_io_ratio",
        "seq_write_ratio", "seq_read_ratio",
        "metadata_time_ratio", "collective_ratio", "total_bw_mb_s",
        "POSIX_WRITES", "POSIX_READS", "POSIX_FSYNCS", "POSIX_OPENS",
        "num_files", "time_imbalance", "byte_imbalance",
    ]
    return {k: round(float(features.get(k, 0)), 4)
            for k in summary_keys if features.get(k, 0) != 0}


def _normalize_result(result, condition, workload_name, has_ml, has_shap, has_kb):
    """Normalize pipeline.analyze() output to standard format."""
    if isinstance(result, dict):
        # Extract from pipeline output format
        gnd = result.get("groundedness", result.get("step4_recommendation", {}).get("groundedness", {}))
        rec = result.get("recommendation", result.get("step4_recommendation", {}).get("parsed"))
        meta = result.get("metadata", result.get("step4_recommendation", {}).get("metadata", {}))
        det = result.get("detected", result.get("step2_detection", {}).get("detected", []))

        return {
            "condition": condition,
            "workload": workload_name,
            "has_ml": has_ml, "has_shap": has_shap, "has_kb": has_kb,
            "detected": det,
            "recommendation": rec,
            "metadata": meta,
            "groundedness": gnd,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Fair ablation study for IOSage")
    parser.add_argument("--n-workloads", type=int, default=8,
                        help="Number of workloads (default: 8)")
    parser.add_argument("--conditions", default="all",
                        help="Comma-separated conditions (C0-C5) or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load env for API keys
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
    test_feat = pd.read_parquet(
        PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet"
    )
    test_labels = pd.read_parquet(
        PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet"
    )

    workload_indices = select_ablation_workloads(test_feat, test_labels, args.n_workloads)
    logger.info("Selected %d workloads for fair ablation", len(workload_indices))

    # Determine which conditions to run
    if args.conditions == "all":
        conditions = ["C0", "C1", "C2", "C3", "C4", "C5"]
    else:
        conditions = [c.strip() for c in args.conditions.split(",")]

    condition_funcs = {
        "C0": run_c0_full,
        "C1": run_c1_no_ml,
        "C2": run_c2_no_kb,
        "C3": run_c3_no_shap,
        "C4": run_c4_ml_only,
        "C5": run_c5_llm_only,
    }

    condition_names = {
        "C0": "Full pipeline",
        "C1": "w/o ML classifier",
        "C2": "w/o knowledge grounding",
        "C3": "w/o feature attribution",
        "C4": "Detection only (no recommendations)",
        "C5": "LLM only (no ML, no KB)",
    }

    all_results = {}

    for cond in conditions:
        func = condition_funcs[cond]
        logger.info("")
        logger.info("=" * 60)
        logger.info("CONDITION %s: %s", cond, condition_names[cond])
        logger.info("=" * 60)

        cond_results = []
        for w_idx, idx in enumerate(workload_indices):
            features = test_feat.iloc[idx].to_dict()
            label_row = test_labels.iloc[idx]

            # Build workload name from labels
            gt_dims = [d for d in BOTTLENECK_DIMS if label_row.get(d, 0) == 1]
            name = f"wl{w_idx}_{'+'.join(gt_dims) if gt_dims else 'healthy'}"

            logger.info("  Workload %d/%d: %s", w_idx + 1, len(workload_indices), name)

            try:
                result = func(pipeline, features, name)
                cond_results.append(result)

                # Log key result
                gnd = result.get("groundedness", {})
                gnd_score = gnd.get("groundedness_score")
                rec = result.get("recommendation")
                n_recs = 0
                if rec and isinstance(rec, dict):
                    n_recs = len(rec.get("recommendations", []))
                logger.info("    → Groundedness=%.3f, #Recs=%d",
                            gnd_score if gnd_score is not None else -1, n_recs)

            except Exception as e:
                logger.error("  FAILED: %s", e, exc_info=True)
                cond_results.append({
                    "condition": cond, "workload": name, "error": str(e)
                })

        all_results[cond] = cond_results

    # =====================================================================
    # Summary Table
    # =====================================================================
    logger.info("")
    logger.info("=" * 90)
    logger.info("FAIR ABLATION RESULTS")
    logger.info("=" * 90)
    logger.info(f"{'Condition':<40s} {'Grnd.':<8s} {'#Recs':<8s} {'Lat.':<10s}")
    logger.info("-" * 66)

    summary = {}
    for cond in conditions:
        results = all_results.get(cond, [])
        valid = [r for r in results if "error" not in r]
        name = condition_names[cond]

        gnd_scores = []
        n_recs_list = []
        latencies = []

        for r in valid:
            g = r.get("groundedness", {})
            gs = g.get("groundedness_score")
            if gs is not None:
                gnd_scores.append(gs)

            rec = r.get("recommendation")
            if rec and isinstance(rec, dict):
                n_recs_list.append(len(rec.get("recommendations", [])))

            m = r.get("metadata", {})
            lat = m.get("latency_ms", 0)
            if lat:
                latencies.append(lat)

        avg_gnd = np.mean(gnd_scores) if gnd_scores else None
        avg_recs = np.mean(n_recs_list) if n_recs_list else None
        avg_lat = np.mean(latencies) if latencies else None

        gnd_str = f"{avg_gnd:.3f}" if avg_gnd is not None else "N/A"
        recs_str = f"{avg_recs:.1f}" if avg_recs is not None else "N/A"
        lat_str = f"{avg_lat/1000:.1f}s" if avg_lat is not None else "<1ms"

        logger.info(f"{name:<40s} {gnd_str:<8s} {recs_str:<8s} {lat_str:<10s}")

        summary[cond] = {
            "name": name,
            "avg_groundedness": float(avg_gnd) if avg_gnd is not None else None,
            "avg_n_recs": float(avg_recs) if avg_recs is not None else None,
            "avg_latency_ms": float(avg_lat) if avg_lat is not None else None,
            "n_workloads": len(valid),
            "n_errors": len(results) - len(valid),
        }

    logger.info("=" * 90)

    # Save results
    results_dir = PROJECT_DIR / "results" / "ablation_fair"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    results_path = results_dir / f"fair_ablation_{timestamp}.json"
    summary_path = results_dir / f"fair_ablation_summary_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Full results: %s", results_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
