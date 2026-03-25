"""
Measure per-stage latency breakdown for the IOPrescriber pipeline.

Pipeline stages:
  1. Feature extraction (parse + extract + engineer)
  2. ML inference (XGBoost predict_proba for all 8 dimensions)
  3. SHAP explanation (TreeExplainer for detected dimensions)
  4. KB retrieval (filter + rank)
  5. LLM cache hit (read + parse cached response)

Reports mean, std, p50, p95, p99 for each stage.
Saves results to results/latency_breakdown.json
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def compute_stats(times_ms):
    """Compute summary statistics for a list of latency measurements."""
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "n_reps": len(arr),
    }


def main():
    seed = 42
    np.random.seed(seed)
    n_reps_fast = 100  # for ML/SHAP/KB/LLM-cache (fast stages)
    n_reps_feat = 10   # for feature extraction (slower, involves I/O)

    # --- Pick a benchmark Darshan log ---
    ior_dir = PROJECT_DIR / "data" / "benchmark_logs" / "ior"
    darshan_files = sorted(ior_dir.glob("*.darshan"))
    if not darshan_files:
        print("ERROR: No .darshan files in data/benchmark_logs/ior/")
        sys.exit(1)
    darshan_path = str(darshan_files[0])
    print(f"Using Darshan log: {Path(darshan_path).name}")

    # =====================================================================
    # Stage 1: Feature extraction (parse + extract + engineer)
    # =====================================================================
    print(f"\n--- Stage 1: Feature Extraction ({n_reps_feat} reps) ---")
    from src.data.parse_darshan import parse_darshan_log
    from src.data.feature_extraction import extract_raw_features
    from src.data.preprocessing import stage3_engineer

    feat_times = []
    features_dict = None
    for i in range(n_reps_feat):
        t0 = time.perf_counter()
        parsed = parse_darshan_log(darshan_path)
        raw_features = extract_raw_features(parsed)
        df = pd.DataFrame([raw_features])
        df = stage3_engineer(df)
        features_dict = df.iloc[0].to_dict()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        feat_times.append(elapsed_ms)
        if i == 0:
            print(f"  Rep 0: {elapsed_ms:.1f} ms, {len(features_dict)} features")

    feat_stats = compute_stats(feat_times)
    print(f"  Feature extraction: mean={feat_stats['mean_ms']:.1f}ms, "
          f"p50={feat_stats['p50_ms']:.1f}ms, p95={feat_stats['p95_ms']:.1f}ms")

    # =====================================================================
    # Stage 2: ML Inference (all 8 XGBoost models)
    # =====================================================================
    print(f"\n--- Stage 2: ML Inference ({n_reps_fast} reps) ---")
    import yaml
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        config = yaml.safe_load(f)

    model_path = PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # Get feature columns (same as Detector._get_feature_cols)
    prod_feat = pd.read_parquet(PROJECT_DIR / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]
    del prod_feat  # free memory

    DIMENSIONS = [
        "access_granularity", "metadata_intensity", "parallelism_efficiency",
        "access_pattern", "interface_choice", "file_strategy",
        "throughput_utilization", "healthy",
    ]

    # Build feature vector
    X = np.array([[features_dict.get(col, 0) for col in feature_cols]], dtype=np.float32)
    print(f"  Feature vector shape: {X.shape}")

    # Warmup
    for dim in DIMENSIONS:
        if dim in models:
            _ = models[dim].predict_proba(X)

    ml_times = []
    predictions = {}
    for i in range(n_reps_fast):
        t0 = time.perf_counter()
        preds = {}
        for dim in DIMENSIONS:
            if dim in models:
                preds[dim] = float(models[dim].predict_proba(X)[0][1])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ml_times.append(elapsed_ms)
        if i == 0:
            predictions = preds

    detected = [d for d in DIMENSIONS
                if predictions.get(d, 0) > 0.3 and d != "healthy"]
    if not detected:
        detected = ["healthy"]

    ml_stats = compute_stats(ml_times)
    print(f"  ML inference: mean={ml_stats['mean_ms']:.1f}ms, "
          f"p50={ml_stats['p50_ms']:.1f}ms, p95={ml_stats['p95_ms']:.1f}ms")
    print(f"  Detected: {detected}")
    print(f"  Predictions: { {d: round(v, 3) for d, v in predictions.items()} }")

    # =====================================================================
    # Stage 3: SHAP Explanation
    # =====================================================================
    print(f"\n--- Stage 3: SHAP Explanation ({n_reps_fast} reps) ---")
    import shap

    # Pre-build TreeExplainers (one-time cost)
    t0 = time.perf_counter()
    explainers = {}
    for dim in DIMENSIONS:
        if dim in models:
            explainers[dim] = shap.TreeExplainer(models[dim])
    explainer_build_ms = (time.perf_counter() - t0) * 1000
    print(f"  TreeExplainer build (one-time): {explainer_build_ms:.1f}ms")

    # Warmup
    for dim in detected:
        if dim in explainers:
            _ = explainers[dim].shap_values(X)

    shap_times = []
    shap_features = {}
    for i in range(n_reps_fast):
        t0 = time.perf_counter()
        attrs = {}
        for dim in detected:
            if dim not in explainers:
                continue
            sv = explainers[dim].shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]
            top_idx = np.argsort(np.abs(sv))[-10:][::-1]
            attrs[dim] = [
                {"feature": feature_cols[idx], "abs_importance": float(abs(sv[idx]))}
                for idx in top_idx
            ]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        shap_times.append(elapsed_ms)
        if i == 0:
            shap_features = attrs

    shap_stats = compute_stats(shap_times)
    print(f"  SHAP explanation: mean={shap_stats['mean_ms']:.1f}ms, "
          f"p50={shap_stats['p50_ms']:.1f}ms, p95={shap_stats['p95_ms']:.1f}ms")
    print(f"  Explained {len(shap_features)} dimensions")

    # =====================================================================
    # Stage 4: KB Retrieval
    # =====================================================================
    print(f"\n--- Stage 4: KB Retrieval ({n_reps_fast} reps) ---")
    from src.ioprescriber.retriever import Retriever

    # One-time KB load
    t0 = time.perf_counter()
    retriever = Retriever()
    kb_load_ms = (time.perf_counter() - t0) * 1000
    print(f"  KB load (one-time): {kb_load_ms:.1f}ms, {len(retriever.kb)} entries")

    kb_times = []
    kb_entries = []
    for i in range(n_reps_fast):
        t0 = time.perf_counter()
        entries = retriever.retrieve(detected, features_dict)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        kb_times.append(elapsed_ms)
        if i == 0:
            kb_entries = entries

    kb_stats = compute_stats(kb_times)
    print(f"  KB retrieval: mean={kb_stats['mean_ms']:.1f}ms, "
          f"p50={kb_stats['p50_ms']:.1f}ms, p95={kb_stats['p95_ms']:.1f}ms")
    print(f"  Retrieved {len(kb_entries)} entries")

    # =====================================================================
    # Stage 5: LLM cache hit latency (read + parse, no API call)
    # =====================================================================
    print(f"\n--- Stage 5: LLM Cache Hit ({n_reps_fast} reps) ---")
    cache_dir = PROJECT_DIR / "data" / "llm_cache" / "ioprescriber"
    cache_files = sorted(cache_dir.glob("*.json"))
    if not cache_files:
        print("  WARNING: No LLM cache files found")
        llm_stats = {"mean_ms": 0, "std_ms": 0, "p50_ms": 0, "p95_ms": 0,
                     "p99_ms": 0, "min_ms": 0, "max_ms": 0, "n_reps": 0,
                     "note": "no cache files"}
    else:
        cache_file = cache_files[0]  # representative file

        # Warmup
        with open(cache_file) as f:
            _ = json.load(f)

        llm_times = []
        for i in range(n_reps_fast):
            t0 = time.perf_counter()
            # Simulate what Recommender.call_llm does on cache hit
            with open(cache_file) as f:
                cached = json.load(f)
            response_text = cached["response"]
            # Parse JSON from response (same as parse_response)
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 3:
                    text = parts[1].strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            elapsed_ms = (time.perf_counter() - t0) * 1000
            llm_times.append(elapsed_ms)

        llm_stats = compute_stats(llm_times)
        print(f"  LLM cache hit: mean={llm_stats['mean_ms']:.1f}ms, "
              f"p50={llm_stats['p50_ms']:.1f}ms, p95={llm_stats['p95_ms']:.1f}ms")
        print(f"  Cache file: {cache_file.name} ({cache_file.stat().st_size / 1024:.1f} KB)")

    # =====================================================================
    # End-to-end summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN SUMMARY")
    print("=" * 60)

    # Compute end-to-end estimates
    ml_only_p50 = feat_stats["p50_ms"] + ml_stats["p50_ms"]
    full_pipeline_p50 = (feat_stats["p50_ms"] + ml_stats["p50_ms"] +
                         shap_stats["p50_ms"] + kb_stats["p50_ms"] +
                         llm_stats["p50_ms"])
    ml_only_p95 = feat_stats["p95_ms"] + ml_stats["p95_ms"]
    full_pipeline_p95 = (feat_stats["p95_ms"] + ml_stats["p95_ms"] +
                         shap_stats["p95_ms"] + kb_stats["p95_ms"] +
                         llm_stats["p95_ms"])

    stages = [
        ("1_feature_extraction", feat_stats),
        ("2_ml_inference", ml_stats),
        ("3_shap_explanation", shap_stats),
        ("4_kb_retrieval", kb_stats),
        ("5_llm_cache_hit", llm_stats),
    ]

    for name, stats in stages:
        print(f"  {name:30s}: mean={stats['mean_ms']:8.1f}ms  "
              f"p50={stats['p50_ms']:8.1f}ms  p95={stats['p95_ms']:8.1f}ms  "
              f"p99={stats['p99_ms']:8.1f}ms")

    print(f"\n  {'ML-only (feat+infer) p50':30s}: {ml_only_p50:.1f}ms")
    print(f"  {'ML-only (feat+infer) p95':30s}: {ml_only_p95:.1f}ms")
    print(f"  {'Full pipeline p50':30s}: {full_pipeline_p50:.1f}ms")
    print(f"  {'Full pipeline p95':30s}: {full_pipeline_p95:.1f}ms")

    # One-time costs
    print(f"\n  One-time initialization costs:")
    print(f"    TreeExplainer build: {explainer_build_ms:.1f}ms")
    print(f"    KB load: {kb_load_ms:.1f}ms")

    # =====================================================================
    # Save results
    # =====================================================================
    results = {
        "darshan_log": Path(darshan_path).name,
        "stages": {name: stats for name, stats in stages},
        "one_time_costs": {
            "tree_explainer_build_ms": round(explainer_build_ms, 1),
            "kb_load_ms": round(kb_load_ms, 1),
        },
        "end_to_end": {
            "ml_only_p50_ms": round(ml_only_p50, 1),
            "ml_only_p95_ms": round(ml_only_p95, 1),
            "full_pipeline_p50_ms": round(full_pipeline_p50, 1),
            "full_pipeline_p95_ms": round(full_pipeline_p95, 1),
        },
        "pipeline_context": {
            "n_models": len(models),
            "n_features": len(feature_cols),
            "n_detected": len(detected),
            "detected_dims": detected,
            "n_kb_entries": len(retriever.kb),
            "n_kb_retrieved": len(kb_entries),
        },
        "config": {
            "seed": seed,
            "n_reps_fast": n_reps_fast,
            "n_reps_feat": n_reps_feat,
        },
    }

    results_path = PROJECT_DIR / "results" / "latency_breakdown.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
