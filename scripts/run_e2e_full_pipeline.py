"""
Run full IOPrescriber pipeline on real Polaris production application logs.

Addresses SC reviewer weakness W3: "No real application diagnosed end-to-end."

Selects representative production jobs (pathological + healthy) and runs the
complete pipeline: ML detect -> SHAP explain -> KB retrieve -> LLM recommend.

Usage:
    source .env && python scripts/run_e2e_full_pipeline.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

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

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def select_candidates(feat, lab, bottleneck_cols):
    """Select representative pathological and healthy production jobs."""
    lab = lab.copy()
    lab["n_bottlenecks"] = lab[bottleneck_cols].sum(axis=1)
    candidates = []

    # --- Candidate 1: Multi-bottleneck job (nprocs >= 4, significant I/O) ---
    multi_bn = lab[lab["n_bottlenecks"] >= 3].index
    for idx in multi_bn:
        nprocs = feat.iloc[idx]["nprocs"]
        bytes_w = feat.iloc[idx]["POSIX_BYTES_WRITTEN"]
        bytes_r = feat.iloc[idx]["POSIX_BYTES_READ"]
        runtime = feat.iloc[idx]["runtime_seconds"]
        if nprocs >= 4 and (bytes_w + bytes_r) > 1e8 and runtime > 10:
            bns = [c for c in bottleneck_cols if lab.iloc[idx][c] == 1]
            candidates.append({
                "idx": idx,
                "name": f"polaris_job_{int(lab.iloc[idx]['_jobid'])}_pathological",
                "description": f"Production job with {len(bns)} bottlenecks: {', '.join(bns)}",
                "category": "pathological",
                "jobid": int(lab.iloc[idx]["_jobid"]),
                "bottlenecks": bns,
            })
            break

    # --- Candidate 2: access_granularity only (common bottleneck) ---
    ag_only = lab[(lab["access_granularity"] == 1) & (lab["n_bottlenecks"] == 1)].index
    for idx in ag_only:
        nprocs = feat.iloc[idx]["nprocs"]
        bytes_w = feat.iloc[idx]["POSIX_BYTES_WRITTEN"]
        runtime = feat.iloc[idx]["runtime_seconds"]
        if nprocs >= 16 and bytes_w > 1e8 and runtime > 30:
            candidates.append({
                "idx": idx,
                "name": f"polaris_job_{int(lab.iloc[idx]['_jobid'])}_small_writes",
                "description": "Production job with small I/O access granularity",
                "category": "single_bottleneck",
                "jobid": int(lab.iloc[idx]["_jobid"]),
                "bottlenecks": ["access_granularity"],
            })
            break

    # --- Candidate 3: throughput_utilization only ---
    tu_only = lab[(lab["throughput_utilization"] == 1) & (lab["n_bottlenecks"] == 1)].index
    for idx in tu_only:
        nprocs = feat.iloc[idx]["nprocs"]
        bytes_w = feat.iloc[idx]["POSIX_BYTES_WRITTEN"]
        runtime = feat.iloc[idx]["runtime_seconds"]
        if nprocs >= 8 and bytes_w > 1e8 and runtime > 30:
            candidates.append({
                "idx": idx,
                "name": f"polaris_job_{int(lab.iloc[idx]['_jobid'])}_low_throughput",
                "description": "Production job with low throughput utilization",
                "category": "single_bottleneck",
                "jobid": int(lab.iloc[idx]["_jobid"]),
                "bottlenecks": ["throughput_utilization"],
            })
            break

    # --- Candidate 4: Healthy job with significant I/O ---
    healthy_mask = lab["healthy"] == 1
    for idx in lab[healthy_mask].index:
        nprocs = feat.iloc[idx]["nprocs"]
        bytes_w = feat.iloc[idx]["POSIX_BYTES_WRITTEN"]
        bytes_r = feat.iloc[idx]["POSIX_BYTES_READ"]
        runtime = feat.iloc[idx]["runtime_seconds"]
        bw = feat.iloc[idx]["total_bw_mb_s"]
        if nprocs >= 16 and (bytes_w + bytes_r) > 1e8 and runtime > 30:
            candidates.append({
                "idx": idx,
                "name": f"polaris_job_{int(lab.iloc[idx]['_jobid'])}_healthy",
                "description": f"Production job with healthy I/O ({bw:.0f} MB/s)",
                "category": "healthy",
                "jobid": int(lab.iloc[idx]["_jobid"]),
                "bottlenecks": ["healthy"],
            })
            break

    return candidates


def main():
    logger.info("Loading production data...")
    feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "production" / "features.parquet")
    lab = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "production" / "labels.parquet")

    bottleneck_cols = [
        "access_granularity", "metadata_intensity", "parallelism_efficiency",
        "access_pattern", "interface_choice", "file_strategy", "throughput_utilization",
    ]

    logger.info("Selecting representative production jobs...")
    candidates = select_candidates(feat, lab, bottleneck_cols)
    logger.info("Selected %d candidates", len(candidates))
    for c in candidates:
        logger.info("  %s: %s", c["name"], c["description"])

    # Initialize the full pipeline
    logger.info("Initializing IOPrescriber pipeline...")
    from src.ioprescriber.pipeline import IOPrescriber
    pipeline = IOPrescriber(llm_model="claude-sonnet")

    all_results = []

    for cand in candidates:
        idx = cand["idx"]
        features_dict = feat.iloc[idx].to_dict()
        label_dict = lab.iloc[idx].to_dict()

        logger.info("")
        logger.info("=" * 70)
        logger.info("ANALYZING: %s", cand["name"])
        logger.info("  Job ID: %d, Category: %s", cand["jobid"], cand["category"])
        logger.info("  Ground truth labels: %s", cand["bottlenecks"])
        logger.info("=" * 70)

        result = pipeline.analyze(features_dict, workload_name=cand["name"])

        # Enrich result with metadata
        result["candidate_info"] = cand
        result["ground_truth"] = {col: int(label_dict.get(col, 0)) for col in bottleneck_cols + ["healthy"]}
        result["job_summary"] = {
            "jobid": cand["jobid"],
            "nprocs": int(features_dict.get("nprocs", 0)),
            "runtime_seconds": round(float(features_dict.get("runtime_seconds", 0)), 1),
            "POSIX_BYTES_WRITTEN": float(features_dict.get("POSIX_BYTES_WRITTEN", 0)),
            "POSIX_BYTES_READ": float(features_dict.get("POSIX_BYTES_READ", 0)),
            "POSIX_WRITES": float(features_dict.get("POSIX_WRITES", 0)),
            "POSIX_READS": float(features_dict.get("POSIX_READS", 0)),
            "avg_write_size": round(float(features_dict.get("avg_write_size", 0)), 1),
            "total_bw_mb_s": round(float(features_dict.get("total_bw_mb_s", 0)), 2),
            "small_io_ratio": round(float(features_dict.get("small_io_ratio", 0)), 4),
            "seq_write_ratio": round(float(features_dict.get("seq_write_ratio", 0)), 4),
            "metadata_time_ratio": round(float(features_dict.get("metadata_time_ratio", 0)), 4),
        }

        # Check ML vs ground truth
        gt_bns = set(c for c in bottleneck_cols if label_dict.get(c, 0) == 1)
        pred_bns = set(result["step1_detection"]["detected"])
        if "healthy" in pred_bns:
            pred_bns.discard("healthy")
        result["ml_vs_gt"] = {
            "ground_truth_bottlenecks": sorted(gt_bns),
            "predicted_bottlenecks": sorted(pred_bns),
            "true_positives": sorted(gt_bns & pred_bns),
            "false_positives": sorted(pred_bns - gt_bns),
            "false_negatives": sorted(gt_bns - pred_bns),
            "correct": gt_bns == pred_bns,
        }

        logger.info("ML vs GT: TP=%s, FP=%s, FN=%s",
                    result["ml_vs_gt"]["true_positives"],
                    result["ml_vs_gt"]["false_positives"],
                    result["ml_vs_gt"]["false_negatives"])

        all_results.append(result)

    # Save results
    output_dir = PROJECT_DIR / "results" / "e2e_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e2e_full_pipeline.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("")
    logger.info("Results saved: %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("E2E FULL PIPELINE RESULTS SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"\n--- {r['workload']} ---")
        print(f"  Category: {r['candidate_info']['category']}")
        print(f"  Job: {r['job_summary']['jobid']}, {r['job_summary']['nprocs']} procs, "
              f"{r['job_summary']['runtime_seconds']}s")
        print(f"  BW: {r['job_summary']['total_bw_mb_s']} MB/s, "
              f"avg write: {r['job_summary']['avg_write_size']:.0f} B")
        print(f"  Ground truth: {r['ml_vs_gt']['ground_truth_bottlenecks']}")
        print(f"  ML detected:  {r['ml_vs_gt']['predicted_bottlenecks']}")
        print(f"  ML correct:   {r['ml_vs_gt']['correct']}")

        # SHAP
        shap = r.get("step2_shap", {})
        for dim, feats in shap.items():
            if feats:
                top = feats[0]
                print(f"  SHAP {dim}: {top['feature']} (|SHAP|={top['abs_importance']:.3f})")

        # KB
        kb = r.get("step3_retrieval", {})
        print(f"  KB entries: {kb.get('n_entries', 0)}")
        for e in kb.get("entries", [])[:2]:
            print(f"    - {e['entry_id']} (sim={e['similarity']:.3f})")

        # LLM
        rec = r.get("step4_recommendation", {})
        if rec.get("parsed"):
            parsed = rec["parsed"]
            print(f"  LLM diagnosis: {parsed.get('diagnosis', 'N/A')[:120]}...")
            n_recs = len(parsed.get("recommendations", []))
            print(f"  Recommendations: {n_recs}")
            for i, rx in enumerate(parsed.get("recommendations", [])[:3]):
                print(f"    {i+1}. [{rx.get('bottleneck_dimension', '?')}] "
                      f"{rx.get('explanation', 'N/A')[:80]}...")
                print(f"       Speedup: {rx.get('expected_speedup', '?')}, "
                      f"Confidence: {rx.get('confidence', '?')}")
            gs = rec.get("groundedness", {})
            print(f"  Groundedness: {gs.get('groundedness_score', 0):.2f} "
                  f"({gs.get('n_grounded', 0)}/{gs.get('n_recommendations', 0)})")
        elif rec.get("groundedness") is None:
            print(f"  LLM: SKIPPED (no API key)")
        else:
            print(f"  LLM: Parse error")

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
