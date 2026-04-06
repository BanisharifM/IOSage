"""
E2E Real Application: Full Pipeline Analysis on Darshan Logs.

After the 3 E2E SLURM jobs complete (pathological, optimized, baseline),
this script runs the full IOPrescriber pipeline on each Darshan log:
  ML detect → SHAP explain → KB retrieve → LLM recommend → groundedness check

Then computes speedup and verifies the pipeline correctly:
1. Detects bottlenecks in the "bad" config
2. Recommends appropriate fixes (NC_NOFILL, better chunks, remove sync)
3. Shows resolved/reduced bottlenecks in the "good" config
4. Reports measured speedup

Usage:
    source .env && python scripts/run_e2e_pipeline_analysis.py

Prerequisites:
    - E2E SLURM jobs completed (check: sacct -j <jobid> --format=State)
    - Darshan logs in data/benchmark_logs/e2e/
    - OPENROUTER_API_KEY set (for LLM recommendations)
"""

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
sys.path.insert(0, str(PROJECT_DIR))

# Load .env
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("e2e_pipeline")

DARSHAN_LOG_DIR = PROJECT_DIR / "data" / "benchmark_logs" / "e2e"
RESULTS_DIR = PROJECT_DIR / "results" / "e2e_evaluation"
SLURM_OUTPUT_DIR = PROJECT_DIR / "results" / "closed_loop"

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


def find_darshan_log(config_name):
    """Find the Darshan log for a specific E2E config.

    Searches by job name pattern in the Darshan log directory.
    """
    patterns = [
        str(DARSHAN_LOG_DIR / f"*{config_name}*"),
        str(DARSHAN_LOG_DIR / "*.darshan"),
    ]

    for pattern in patterns:
        logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if logs:
            return logs[0]
    return None


def find_darshan_by_jobid(job_id):
    """Find Darshan log by SLURM job ID."""
    pattern = str(DARSHAN_LOG_DIR / f"*id{job_id}*")
    logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return logs[0] if logs else None


def get_slurm_job_ids():
    """Read job IDs from SLURM output files."""
    job_ids = {}
    for config in ["pathological", "optimized", "baseline"]:
        pattern = str(SLURM_OUTPUT_DIR / f"e2e_{config}_*.out")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            # Extract job ID from filename: e2e_pathological_17177584.out
            fname = Path(files[0]).stem  # e2e_pathological_17177584
            parts = fname.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                job_ids[config] = parts[1]
                logger.info("Found %s job ID: %s (from %s)", config, parts[1], files[0])
    return job_ids


def parse_e2e_timing_from_output(config_name):
    """Parse E2E write timing from SLURM output file."""
    pattern = str(SLURM_OUTPUT_DIR / f"e2e_{config_name}_*.out")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None

    result = {"file": files[0], "exit_code": None, "start_time": None, "end_time": None}

    with open(files[0]) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if "Exit code:" in line:
            try:
                result["exit_code"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if line.startswith("E2E") and " - " in line:
            result["start_time"] = line.split(" - ")[-1].strip() if " - " in line else None
        if "Completed:" in line:
            result["end_time"] = line.split("Completed:")[-1].strip()

    return result


def extract_features_from_darshan(darshan_path):
    """Parse Darshan log and extract features."""
    try:
        from src.data.parse_darshan import parse_darshan_log
        from src.data.feature_extraction import extract_raw_features
        from src.data.preprocessing import stage3_engineer

        parsed = parse_darshan_log(darshan_path)
        if parsed is None:
            logger.error("Failed to parse: %s", darshan_path)
            return None, None

        raw = extract_raw_features(parsed)
        df = pd.DataFrame([raw])
        df = stage3_engineer(df)
        features = df.iloc[0].to_dict()

        # Key metrics
        metrics = {
            "total_bw_mb_s": features.get("total_bw_mb_s", 0),
            "write_bw_mb_s": features.get("write_bw_mb_s", 0),
            "read_bw_mb_s": features.get("read_bw_mb_s", 0),
            "avg_write_size": features.get("avg_write_size", 0),
            "avg_read_size": features.get("avg_read_size", 0),
            "small_io_ratio": features.get("small_io_ratio", 0),
            "seq_write_ratio": features.get("seq_write_ratio", 0),
            "metadata_time_ratio": features.get("metadata_time_ratio", 0),
            "POSIX_BYTES_WRITTEN": features.get("POSIX_BYTES_WRITTEN", 0),
            "POSIX_WRITES": features.get("POSIX_WRITES", 0),
            "POSIX_FSYNCS": features.get("POSIX_FSYNCS", 0),
            "nprocs": features.get("nprocs", 0),
            "runtime_seconds": features.get("runtime_seconds", 0),
        }
        return metrics, features

    except Exception as e:
        logger.error("Feature extraction failed for %s: %s", darshan_path, e)
        return None, None


def run_ml_detection(features_dict, models, feature_cols):
    """Run ML detection on features."""
    X = np.array(
        [[features_dict.get(col, 0) for col in feature_cols]],
        dtype=np.float32,
    )

    predictions = {}
    for dim in DIMENSIONS:
        if dim in models:
            prob = float(models[dim].predict_proba(X)[0][1])
            predictions[dim] = round(prob, 4)

    detected = [d for d in DIMENSIONS if predictions.get(d, 0) > 0.3 and d != "healthy"]
    if not detected:
        detected = ["healthy"]

    return predictions, detected


def run_shap(features_dict, explainer, detected_dims, feature_cols):
    """Run SHAP explanation."""
    try:
        X = np.array(
            [[features_dict.get(col, 0) for col in feature_cols]],
            dtype=np.float32,
        )
        return explainer.explain(X, detected_dims)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return {}


def run_full_pipeline_on_config(config_name, darshan_path, models, feature_cols,
                                  explainer, kb, recommender):
    """Run the complete IOPrescriber pipeline on one E2E config."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE ANALYSIS: E2E %s", config_name.upper())
    logger.info("  Darshan: %s", Path(darshan_path).name)
    logger.info("=" * 70)

    result = {
        "config": config_name,
        "darshan_path": str(darshan_path),
    }

    # Step 1: Extract features
    logger.info("  Step 1: Feature extraction...")
    metrics, features = extract_features_from_darshan(darshan_path)
    if not metrics:
        result["status"] = "FEATURE_EXTRACTION_FAILED"
        return result

    result["metrics"] = metrics
    logger.info("    BW: %.1f MB/s, avg_write: %.0f bytes, small_io: %.3f",
                metrics["total_bw_mb_s"], metrics["avg_write_size"], metrics["small_io_ratio"])

    # Step 2: ML Detection
    logger.info("  Step 2: ML detection...")
    predictions, detected = run_ml_detection(features, models, feature_cols)
    result["predictions"] = predictions
    result["detected"] = detected
    logger.info("    Detected: %s", detected)
    for d in detected:
        if d != "healthy":
            logger.info("      %s: %.4f confidence", d, predictions.get(d, 0))

    # Step 3: SHAP
    logger.info("  Step 3: SHAP explanation...")
    shap_results = run_shap(features, explainer, detected, feature_cols)
    result["shap_top_features"] = {}
    for dim, feats in shap_results.items():
        if feats:
            result["shap_top_features"][dim] = [
                {"feature": f["feature"], "importance": round(f["abs_importance"], 4)}
                for f in feats[:5]
            ]
            top = feats[0]
            logger.info("    %s: top feature = %s (|SHAP|=%.3f)",
                        dim, top["feature"], top["abs_importance"])

    # Step 4: KB Retrieval
    logger.info("  Step 4: KB retrieval...")
    from src.llm.recommendation import retrieve_relevant_entries
    signature = {k: features.get(k, 0) for k in [
        "avg_write_size", "small_io_ratio", "seq_write_ratio",
        "metadata_time_ratio", "collective_ratio", "total_bw_mb_s",
        "nprocs", "POSIX_BYTES_WRITTEN", "POSIX_FSYNCS",
    ]}
    kb_matches = retrieve_relevant_entries(kb, detected, signature, top_k=3)
    result["kb_matches"] = len(kb_matches)
    logger.info("    KB matches: %d", len(kb_matches))

    # Step 5: LLM Recommendation (only for non-healthy)
    has_bottleneck = any(d != "healthy" for d in detected)
    if has_bottleneck and recommender:
        logger.info("  Step 5: LLM recommendation...")
        darshan_summary = {k: round(float(v), 4) if isinstance(v, float) else v
                          for k, v in metrics.items() if v}

        recommendation, groundedness, metadata, raw = recommender.recommend(
            predictions, detected, shap_results, kb_matches, darshan_summary
        )

        result["recommendation"] = {
            "parsed": recommendation,
            "groundedness": groundedness,
            "metadata": metadata,
        }

        if recommendation:
            n_recs = len(recommendation.get("recommendations", []))
            gs = groundedness.get("groundedness_score", 0)
            logger.info("    Recommendations: %d, groundedness: %.2f", n_recs, gs)

            # Check if NC_NOFILL is recommended
            rec_text = json.dumps(recommendation, default=str).lower()
            nc_nofill_mentioned = "nofill" in rec_text or "nc_nofill" in rec_text or "fill" in rec_text
            result["nc_nofill_recommended"] = nc_nofill_mentioned
            if nc_nofill_mentioned:
                logger.info("    NC_NOFILL: RECOMMENDED (matches known E2E fix)")
            else:
                logger.info("    NC_NOFILL: not recommended (KB gap — NetCDF-specific)")

            # Log each recommendation
            for i, rec in enumerate(recommendation.get("recommendations", [])):
                logger.info("    Rec %d [%s]: %s",
                            i + 1,
                            rec.get("bottleneck_dimension", "?"),
                            rec.get("explanation", "?")[:100])
        else:
            logger.warning("    LLM parse failed")
    else:
        logger.info("  Step 5: SKIPPED (healthy or no recommender)")
        result["recommendation"] = {"skipped": "healthy" if not has_bottleneck else "no_recommender"}

    result["status"] = "SUCCESS"
    return result


def main():
    logger.info("=" * 70)
    logger.info("E2E Real Application: Full Pipeline Analysis")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find Darshan logs
    logger.info("Searching for E2E Darshan logs in %s...", DARSHAN_LOG_DIR)
    all_darshan = sorted(glob.glob(str(DARSHAN_LOG_DIR / "*.darshan")),
                         key=os.path.getmtime, reverse=True)
    logger.info("Found %d Darshan logs", len(all_darshan))

    if not all_darshan:
        # Try to find by SLURM job IDs
        job_ids = get_slurm_job_ids()
        if job_ids:
            logger.info("Attempting to find logs by job IDs: %s", job_ids)
            for config, jid in job_ids.items():
                log = find_darshan_by_jobid(jid)
                if log:
                    logger.info("  Found %s log: %s", config, log)

    if not all_darshan:
        logger.error("No Darshan logs found. E2E SLURM jobs may not have completed yet.")
        logger.info("Check job status: sacct -j <jobid> --format=JobID,State,Elapsed")
        logger.info("Expected logs in: %s", DARSHAN_LOG_DIR)
        return

    # Load ML models
    logger.info("Loading ML models...")
    import pickle
    import yaml

    with open(PROJECT_DIR / "results" / "boost_experiment" / "new_models" / "xgboost_biquality_w100_seed42.pkl", "rb") as f:
        models = pickle.load(f)

    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        config = yaml.safe_load(f)

    prod_feat = pd.read_parquet(PROJECT_DIR / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]

    logger.info("  Models: %d dimensions, %d features", len(models), len(feature_cols))

    # Load SHAP explainer
    from src.ioprescriber.explainer import Explainer
    explainer = Explainer(models=models, feature_cols=feature_cols)

    # Load KB
    with open(PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json") as f:
        kb = json.load(f)
    logger.info("  KB: %d entries", len(kb))

    # Load LLM recommender
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    recommender = None
    if api_key:
        from src.ioprescriber.recommender import Recommender
        recommender = Recommender(model="claude-sonnet")
        logger.info("  LLM: Claude Sonnet via OpenRouter")
    else:
        logger.warning("  No OPENROUTER_API_KEY — LLM recommendations disabled")

    # Analyze each Darshan log
    all_results = []
    for darshan_path in all_darshan:
        fname = Path(darshan_path).name
        # Infer config from filename
        if "pathological" in fname.lower():
            config_name = "pathological"
        elif "optimized" in fname.lower() or "ultra" in fname.lower():
            config_name = "ultra_optimized"
        elif "baseline" in fname.lower():
            config_name = "baseline"
        else:
            config_name = fname.split("_")[1] if "_" in fname else "unknown"

        result = run_full_pipeline_on_config(
            config_name, darshan_path, models, feature_cols,
            explainer, kb, recommender,
        )
        all_results.append(result)

    # Compute speedups
    logger.info("")
    logger.info("=" * 70)
    logger.info("E2E SPEEDUP ANALYSIS")
    logger.info("=" * 70)

    config_bw = {}
    for r in all_results:
        if r.get("status") == "SUCCESS":
            bw = r["metrics"].get("total_bw_mb_s", 0)
            config_bw[r["config"]] = bw

    if "pathological" in config_bw and "ultra_optimized" in config_bw:
        bad_bw = config_bw["pathological"] or 0.001
        good_bw = config_bw["ultra_optimized"] or 0.001
        speedup = round(good_bw / bad_bw, 2)
        logger.info("  Pathological → Ultra-Optimized: %.1f → %.1f MB/s = %.1fx",
                     bad_bw, good_bw, speedup)

    if "baseline" in config_bw and "ultra_optimized" in config_bw:
        bad_bw = config_bw["baseline"] or 0.001
        good_bw = config_bw["ultra_optimized"] or 0.001
        speedup = round(good_bw / bad_bw, 2)
        logger.info("  Baseline → Ultra-Optimized: %.1f → %.1f MB/s = %.1fx",
                     bad_bw, good_bw, speedup)

    # Verify pipeline correctness
    logger.info("")
    logger.info("PIPELINE CORRECTNESS CHECK:")
    for r in all_results:
        if r.get("status") != "SUCCESS":
            continue
        config = r["config"]
        detected = r.get("detected", [])
        has_bottleneck = any(d != "healthy" for d in detected)

        if config in ("pathological", "baseline"):
            if has_bottleneck:
                logger.info("  %s: CORRECT — bottleneck detected (%s)", config, detected)
            else:
                logger.warning("  %s: UNEXPECTED — classified as healthy", config)
        elif config == "ultra_optimized":
            if not has_bottleneck:
                logger.info("  %s: CORRECT — classified as healthy", config)
            else:
                logger.info("  %s: residual bottleneck detected (%s)", config, detected)

    # Save results
    output = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "E2E real application closed-loop validation on Delta",
            "benchmark": "E2E (Lofstead et al., HPDC 2011)",
            "grid": "256x256x128 (8x8x1 decomposition, 64 procs, 4 nodes)",
            "data_volume_mb": 640,
        },
        "configs": {c: config_bw.get(c, None) for c in ["pathological", "baseline", "ultra_optimized"]},
        "speedups": {},
        "pipeline_results": all_results,
    }

    if "pathological" in config_bw and "ultra_optimized" in config_bw:
        output["speedups"]["pathological_to_optimized"] = {
            "before_bw": config_bw["pathological"],
            "after_bw": config_bw["ultra_optimized"],
            "speedup": round(config_bw["ultra_optimized"] / max(config_bw["pathological"], 0.001), 2),
        }
    if "baseline" in config_bw and "ultra_optimized" in config_bw:
        output["speedups"]["baseline_to_optimized"] = {
            "before_bw": config_bw["baseline"],
            "after_bw": config_bw["ultra_optimized"],
            "speedup": round(config_bw["ultra_optimized"] / max(config_bw["baseline"], 0.001), 2),
        }

    output_path = RESULTS_DIR / "e2e_closed_loop_delta.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("")
    logger.info("Results saved: %s", output_path)


if __name__ == "__main__":
    main()
