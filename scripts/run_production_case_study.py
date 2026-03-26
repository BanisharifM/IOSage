"""
Production Case Study: Run IOPrescriber on 50 Randomly Selected Polaris Logs.

Addresses the "cherry-picking" concern by sampling 50 random production logs
with a fixed seed (42) and running the full IOPrescriber pipeline on each.

Stratified sampling: ~25 logs with at least one heuristic-labeled bottleneck,
~25 logs labeled healthy. This gives a representative cross-section.

Since production logs only have Drishti heuristic labels (which the ML was
partially trained on), we do NOT report "accuracy vs Drishti" as that would
be circular. Instead we report:
  - Distribution of ML-detected bottleneck types
  - Groundedness of LLM recommendations
  - Number of recommendations per job
  - Representative case studies

LLM is only called for non-healthy detections (saves API costs).

Usage:
    source .env && python scripts/run_production_case_study.py

Output:
    results/production_case_study/random_50_results.json
    results/production_case_study/random_50_summary.md
"""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Load .env for OPENROUTER_API_KEY
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
logger = logging.getLogger("production_case_study")

OUTPUT_DIR = PROJECT_DIR / "results" / "production_case_study"

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]
BOTTLENECK_DIMS = [d for d in DIMENSIONS if d != "healthy"]

SEED = 42
N_TOTAL = 50
N_BOTTLENECK = 25
N_HEALTHY = 25


def sample_stratified(features, labels, seed=SEED):
    """Sample 25 bottleneck + 25 healthy logs with fixed seed."""
    rng = np.random.RandomState(seed)

    has_bottleneck = labels[BOTTLENECK_DIMS].max(axis=1) > 0
    bottleneck_idx = labels.index[has_bottleneck].tolist()
    healthy_idx = labels.index[~has_bottleneck].tolist()

    logger.info("Pool: %d bottleneck, %d healthy", len(bottleneck_idx), len(healthy_idx))

    sampled_bottleneck = rng.choice(bottleneck_idx, size=N_BOTTLENECK, replace=False).tolist()
    sampled_healthy = rng.choice(healthy_idx, size=N_HEALTHY, replace=False).tolist()

    all_sampled = sampled_bottleneck + sampled_healthy
    rng.shuffle(all_sampled)

    return all_sampled, sampled_bottleneck, sampled_healthy


def build_darshan_summary(features_row):
    """Build concise Darshan summary dict from feature row."""
    summary_keys = [
        "nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ",
        "avg_write_size", "avg_read_size", "small_io_ratio", "seq_write_ratio",
        "seq_read_ratio", "metadata_time_ratio", "collective_ratio",
        "total_bw_mb_s", "fsync_ratio", "has_mpiio", "num_files",
    ]
    summary = {}
    for k in summary_keys:
        val = features_row.get(k, 0)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            summary[k] = round(float(val), 4)
    return summary


def main():
    logger.info("=" * 70)
    logger.info("Production Case Study: 50 Random Polaris Logs")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading production features and labels...")
    features = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "production" / "features.parquet")
    labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "production" / "labels.parquet")
    logger.info("Loaded %d production logs (%d features)", len(features), features.shape[1])

    # Stratified sampling
    sampled_indices, bottleneck_sample, healthy_sample = sample_stratified(features, labels)
    logger.info("Sampled %d logs: %d bottleneck, %d healthy",
                len(sampled_indices), len(bottleneck_sample), len(healthy_sample))

    # Initialize pipeline
    logger.info("Initializing IOPrescriber pipeline...")
    from src.ioprescriber.pipeline import IOPrescriber
    pipeline = IOPrescriber(llm_model="claude-sonnet")

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set. LLM calls will fail for non-healthy logs.")

    # Process each sample
    all_results = []
    detection_counter = Counter()
    total_recs = 0
    total_grounded = 0
    total_rec_count = 0
    llm_call_count = 0
    llm_skip_count = 0
    total_latency_ms = 0

    for i, idx in enumerate(sampled_indices):
        feat_row = features.loc[idx]
        label_row = labels.loc[idx]
        heuristic_labels = [d for d in BOTTLENECK_DIMS if label_row.get(d, 0) == 1]
        is_heuristic_healthy = len(heuristic_labels) == 0

        job_id = label_row.get("_jobid", f"idx_{idx}")

        logger.info("")
        logger.info("-" * 60)
        logger.info("Sample %d/%d (idx=%d, jobid=%s)", i + 1, N_TOTAL, idx, job_id)
        logger.info("  Heuristic labels: %s", heuristic_labels if heuristic_labels else ["healthy"])

        features_dict = feat_row.to_dict()
        # Clean NaN values
        for k, v in features_dict.items():
            if isinstance(v, float) and np.isnan(v):
                features_dict[k] = 0.0

        darshan_summary = build_darshan_summary(features_dict)

        result = {
            "sample_index": i,
            "dataframe_index": int(idx),
            "job_id": str(job_id),
            "heuristic_labels": heuristic_labels if heuristic_labels else ["healthy"],
            "is_heuristic_healthy": is_heuristic_healthy,
            "darshan_summary": darshan_summary,
        }

        # Step 1: ML Detection
        try:
            predictions, detected = pipeline.detector.detect_from_features(features_dict)
        except Exception as exc:
            logger.error("  Detection failed: %s", exc)
            result["status"] = "DETECTION_FAILED"
            result["error"] = str(exc)
            all_results.append(result)
            continue

        result["ml_predictions"] = predictions
        result["ml_detected"] = detected
        logger.info("  ML detected: %s", detected)

        for d in detected:
            detection_counter[d] += 1

        # Step 2: SHAP (always run, fast)
        try:
            X = np.array([[features_dict.get(col, 0) for col in pipeline.detector.feature_cols]],
                          dtype=np.float32)
            shap_features = pipeline.explainer.explain(X, detected_dims=detected)
            result["shap_top_features"] = {}
            for dim, feats in shap_features.items():
                if feats:
                    result["shap_top_features"][dim] = [
                        {"feature": f["feature"], "importance": round(f["abs_importance"], 4)}
                        for f in feats[:3]
                    ]
        except Exception as exc:
            logger.warning("  SHAP failed: %s", exc)
            shap_features = {}
            result["shap_top_features"] = {}

        # Step 3: KB Retrieval (always run, no API cost)
        try:
            kb_entries = pipeline.retriever.retrieve(detected, features_dict)
            result["kb_retrieval"] = {
                "n_entries": len(kb_entries),
                "entries": [{"entry_id": m["entry"]["entry_id"],
                             "similarity": m["similarity"],
                             "matched_dims": m["matched_dims"]}
                            for m in kb_entries],
            }
        except Exception as exc:
            logger.warning("  KB retrieval failed: %s", exc)
            kb_entries = []
            result["kb_retrieval"] = {"n_entries": 0, "entries": []}

        # Step 4: LLM Recommendation (only for non-healthy detections)
        ml_has_bottleneck = any(d != "healthy" for d in detected)

        if ml_has_bottleneck and api_key:
            logger.info("  Step 4: LLM recommendation...")
            try:
                recommendation, groundedness, metadata, raw_response = pipeline.recommender.recommend(
                    predictions, detected, shap_features, kb_entries, darshan_summary
                )

                result["llm_recommendation"] = {
                    "parsed": recommendation,
                    "groundedness": groundedness,
                    "metadata": metadata,
                }

                if recommendation:
                    n_recs = len(recommendation.get("recommendations", []))
                    total_recs += n_recs
                    total_rec_count += 1
                    total_grounded += groundedness.get("n_grounded", 0)
                    logger.info("  LLM: %d recs, groundedness=%.2f",
                                n_recs, groundedness.get("groundedness_score", 0))
                else:
                    logger.warning("  LLM: parse failed")

                if metadata:
                    total_latency_ms += metadata.get("latency_ms", 0)

                llm_call_count += 1

            except Exception as exc:
                logger.error("  LLM failed: %s", exc)
                result["llm_recommendation"] = {"error": str(exc)}
        else:
            if not ml_has_bottleneck:
                logger.info("  Step 4: SKIPPED (healthy, no recommendation needed)")
                result["llm_recommendation"] = {"skipped": "healthy_detection"}
                llm_skip_count += 1
            else:
                logger.info("  Step 4: SKIPPED (no API key)")
                result["llm_recommendation"] = {"skipped": "no_api_key"}

        result["status"] = "SUCCESS"
        all_results.append(result)

    # Aggregate statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 70)

    # Detection distribution
    logger.info("")
    logger.info("ML Detection Distribution (across %d logs):", N_TOTAL)
    for dim in DIMENSIONS:
        count = detection_counter.get(dim, 0)
        pct = 100 * count / N_TOTAL
        logger.info("  %-28s %3d (%5.1f%%)", dim, count, pct)

    # Multi-label statistics
    n_multi = sum(1 for r in all_results if r.get("status") == "SUCCESS"
                  and len([d for d in r.get("ml_detected", []) if d != "healthy"]) > 1)
    n_single = sum(1 for r in all_results if r.get("status") == "SUCCESS"
                   and len([d for d in r.get("ml_detected", []) if d != "healthy"]) == 1)
    n_healthy_det = detection_counter.get("healthy", 0)

    logger.info("")
    logger.info("Detection patterns:")
    logger.info("  Healthy (no bottleneck): %d", n_healthy_det)
    logger.info("  Single bottleneck:       %d", n_single)
    logger.info("  Multi-label bottleneck:  %d", n_multi)

    # Average bottleneck dimensions per non-healthy log
    bottleneck_counts = [len([d for d in r.get("ml_detected", []) if d != "healthy"])
                         for r in all_results
                         if r.get("status") == "SUCCESS"
                         and any(d != "healthy" for d in r.get("ml_detected", []))]
    if bottleneck_counts:
        logger.info("  Avg dims per bottleneck job: %.1f", np.mean(bottleneck_counts))

    # Groundedness
    logger.info("")
    logger.info("LLM Recommendation Statistics:")
    logger.info("  LLM calls made:       %d", llm_call_count)
    logger.info("  LLM calls skipped:    %d (healthy)", llm_skip_count)

    gs_scores = []
    all_n_recs = []
    for r in all_results:
        rec = r.get("llm_recommendation", {})
        gs = rec.get("groundedness", {})
        if gs and "groundedness_score" in gs:
            gs_scores.append(gs["groundedness_score"])
            all_n_recs.append(gs.get("n_recommendations", 0))

    if gs_scores:
        logger.info("  Mean groundedness:     %.3f", np.mean(gs_scores))
        logger.info("  Median groundedness:   %.3f", np.median(gs_scores))
        logger.info("  Fully grounded (1.0):  %d/%d", sum(1 for s in gs_scores if s >= 1.0), len(gs_scores))
        logger.info("  Total recommendations: %d", sum(all_n_recs))
        logger.info("  Avg recs per job:      %.1f", np.mean(all_n_recs) if all_n_recs else 0)

    # ML vs heuristic agreement (informational only, not "accuracy")
    agree_healthy = 0
    agree_bottleneck = 0
    disagree_h2b = 0  # heuristic=healthy, ML=bottleneck
    disagree_b2h = 0  # heuristic=bottleneck, ML=healthy
    dim_agreement = defaultdict(lambda: {"agree": 0, "disagree": 0})

    for r in all_results:
        if r.get("status") != "SUCCESS":
            continue
        ml_has_bn = any(d != "healthy" for d in r.get("ml_detected", []))
        h_has_bn = not r.get("is_heuristic_healthy", True)

        if not ml_has_bn and not h_has_bn:
            agree_healthy += 1
        elif ml_has_bn and h_has_bn:
            agree_bottleneck += 1
        elif not h_has_bn and not ml_has_bn:
            disagree_b2h += 1
        else:
            disagree_h2b += 1

    logger.info("")
    logger.info("ML vs Heuristic Agreement (NOT accuracy -- informational):")
    logger.info("  Both healthy:     %d", agree_healthy)
    logger.info("  Both bottleneck:  %d", agree_bottleneck)
    logger.info("  Heuristic=H, ML=B: %d", disagree_h2b)
    logger.info("  Heuristic=B, ML=H: %d", disagree_b2h)
    total_agree = agree_healthy + agree_bottleneck
    logger.info("  Agreement rate:   %.1f%% (%d/%d)",
                100 * total_agree / N_TOTAL, total_agree, N_TOTAL)

    # Save results JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "random_50_results.json"

    summary_stats = {
        "n_samples": N_TOTAL,
        "n_bottleneck_sample": N_BOTTLENECK,
        "n_healthy_sample": N_HEALTHY,
        "seed": SEED,
        "detection_distribution": {dim: detection_counter.get(dim, 0) for dim in DIMENSIONS},
        "detection_pct": {dim: round(100 * detection_counter.get(dim, 0) / N_TOTAL, 1) for dim in DIMENSIONS},
        "n_healthy_detected": n_healthy_det,
        "n_single_bottleneck": n_single,
        "n_multi_label": n_multi,
        "avg_dims_per_bottleneck": round(float(np.mean(bottleneck_counts)), 2) if bottleneck_counts else 0,
        "llm_calls_made": llm_call_count,
        "llm_calls_skipped_healthy": llm_skip_count,
        "mean_groundedness": round(float(np.mean(gs_scores)), 4) if gs_scores else 0,
        "median_groundedness": round(float(np.median(gs_scores)), 4) if gs_scores else 0,
        "fully_grounded_count": sum(1 for s in gs_scores if s >= 1.0) if gs_scores else 0,
        "total_recommendations": sum(all_n_recs) if all_n_recs else 0,
        "avg_recs_per_job": round(float(np.mean(all_n_recs)), 2) if all_n_recs else 0,
        "agreement_both_healthy": agree_healthy,
        "agreement_both_bottleneck": agree_bottleneck,
        "disagree_heuristic_h_ml_b": disagree_h2b,
        "disagree_heuristic_b_ml_h": disagree_b2h,
        "agreement_rate_pct": round(100 * total_agree / N_TOTAL, 1),
    }

    final_output = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "IOPrescriber on 50 randomly sampled production Polaris logs",
            "seed": SEED,
            "n_total_production": len(features),
            "llm_model": "anthropic/claude-sonnet-4 (via OpenRouter)",
        },
        "summary_statistics": summary_stats,
        "per_sample_results": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2, default=str)
    logger.info("")
    logger.info("Results saved: %s", results_path)

    # Generate summary markdown
    summary_md = generate_summary_md(summary_stats, all_results)
    summary_path = OUTPUT_DIR / "random_50_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_md)
    logger.info("Summary saved: %s", summary_path)


def generate_summary_md(stats, all_results):
    """Generate a markdown summary of the case study results."""
    lines = []
    lines.append("# Production Case Study: 50 Random Polaris Logs")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Sample size**: {stats['n_samples']} logs (stratified: {stats['n_bottleneck_sample']} bottleneck + {stats['n_healthy_sample']} healthy)")
    lines.append(f"- **Random seed**: {stats['seed']}")
    lines.append(f"- **Source**: 131,151 production Polaris Darshan logs")
    lines.append(f"- **LLM**: Claude Sonnet via OpenRouter (temperature=0)")
    lines.append(f"- **Purpose**: Unbiased accuracy estimate (no cherry-picking)")
    lines.append("")

    lines.append("## ML Detection Distribution")
    lines.append("")
    lines.append("| Bottleneck Dimension | Count | Percentage |")
    lines.append("|---------------------|-------|------------|")
    for dim in DIMENSIONS:
        count = stats["detection_distribution"].get(dim, 0)
        pct = stats["detection_pct"].get(dim, 0)
        lines.append(f"| {dim} | {count} | {pct}% |")
    lines.append("")
    lines.append(f"- Healthy (no bottleneck): {stats['n_healthy_detected']}")
    lines.append(f"- Single bottleneck: {stats['n_single_bottleneck']}")
    lines.append(f"- Multi-label: {stats['n_multi_label']}")
    lines.append(f"- Avg dimensions per bottleneck job: {stats['avg_dims_per_bottleneck']}")
    lines.append("")

    lines.append("## LLM Recommendation Quality")
    lines.append("")
    lines.append(f"- LLM calls made: {stats['llm_calls_made']} (skipped {stats['llm_calls_skipped_healthy']} healthy)")
    lines.append(f"- Mean groundedness: {stats['mean_groundedness']:.3f}")
    lines.append(f"- Median groundedness: {stats['median_groundedness']:.3f}")
    lines.append(f"- Fully grounded (1.0): {stats['fully_grounded_count']}/{stats['llm_calls_made']}")
    lines.append(f"- Total recommendations generated: {stats['total_recommendations']}")
    lines.append(f"- Average recommendations per job: {stats['avg_recs_per_job']:.1f}")
    lines.append("")

    lines.append("## ML vs Heuristic Agreement")
    lines.append("")
    lines.append("Note: This is NOT accuracy -- ML was partially trained on heuristic labels.")
    lines.append("High agreement indicates consistency; disagreements may indicate ML generalization.")
    lines.append("")
    lines.append(f"- Both healthy: {stats['agreement_both_healthy']}")
    lines.append(f"- Both bottleneck: {stats['agreement_both_bottleneck']}")
    lines.append(f"- Heuristic=healthy, ML=bottleneck: {stats['disagree_heuristic_h_ml_b']}")
    lines.append(f"- Heuristic=bottleneck, ML=healthy: {stats['disagree_heuristic_b_ml_h']}")
    lines.append(f"- Overall agreement: {stats['agreement_rate_pct']}%")
    lines.append("")

    # Representative case studies
    lines.append("## Representative Case Studies")
    lines.append("")

    # Find interesting cases: multi-label, high groundedness, disagreements
    interesting = []
    for r in all_results:
        if r.get("status") != "SUCCESS":
            continue

        ml_detected = r.get("ml_detected", [])
        n_bottleneck_dims = len([d for d in ml_detected if d != "healthy"])
        has_llm = r.get("llm_recommendation", {}).get("parsed") is not None
        gs = r.get("llm_recommendation", {}).get("groundedness", {})
        gs_score = gs.get("groundedness_score", 0) if gs else 0

        is_disagreement = (r.get("is_heuristic_healthy") and n_bottleneck_dims > 0) or \
                          (not r.get("is_heuristic_healthy") and n_bottleneck_dims == 0)

        score = n_bottleneck_dims * 2 + (1 if is_disagreement else 0) + (gs_score if has_llm else 0)
        interesting.append((score, r))

    interesting.sort(key=lambda x: x[0], reverse=True)

    for case_num, (_, r) in enumerate(interesting[:5], 1):
        lines.append(f"### Case {case_num}: Job {r.get('job_id', '?')}")
        lines.append("")

        summary = r.get("darshan_summary", {})
        lines.append(f"- **nprocs**: {summary.get('nprocs', '?')}")
        lines.append(f"- **runtime**: {summary.get('runtime_seconds', '?')}s")
        bw = summary.get("total_bw_mb_s", 0)
        lines.append(f"- **bandwidth**: {bw:.1f} MB/s")
        lines.append(f"- **small_io_ratio**: {summary.get('small_io_ratio', 0):.3f}")
        lines.append(f"- **heuristic labels**: {r.get('heuristic_labels', [])}")
        lines.append(f"- **ML detected**: {r.get('ml_detected', [])}")

        # SHAP
        shap = r.get("shap_top_features", {})
        if shap:
            lines.append(f"- **Top SHAP features**:")
            for dim, feats in shap.items():
                if feats:
                    feat_str = ", ".join(f"{f['feature']} ({f['importance']:.3f})" for f in feats[:2])
                    lines.append(f"  - {dim}: {feat_str}")

        # LLM
        rec = r.get("llm_recommendation", {})
        parsed = rec.get("parsed")
        if parsed:
            gs = rec.get("groundedness", {})
            lines.append(f"- **LLM groundedness**: {gs.get('groundedness_score', 0):.2f} ({gs.get('n_grounded', 0)}/{gs.get('n_recommendations', 0)})")
            lines.append(f"- **Diagnosis**: {parsed.get('diagnosis', 'N/A')[:200]}")
            for rx in parsed.get("recommendations", [])[:2]:
                lines.append(f"  - [{rx.get('bottleneck_dimension', '?')}] {rx.get('explanation', 'N/A')[:120]}")
        elif "skipped" in rec:
            lines.append(f"- **LLM**: Skipped ({rec.get('skipped', '')})")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
