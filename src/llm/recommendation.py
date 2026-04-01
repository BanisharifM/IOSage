"""
IOSage Single-Shot RAG-Grounded LLM Recommendation Pipeline.

Single-shot (non-iterative) pipeline that:
1. Takes ML detection results + SHAP features for a Darshan log
2. Retrieves matching KB entries via similarity search
3. Constructs structured prompt with grounded evidence
4. Calls LLM to generate code-level recommendations
5. Verifies groundedness of the response
6. Caches all inputs/outputs for reproducibility

This is the BACKUP pipeline — works without iteration, provides
benchmark-grounded recommendations with source code evidence.

Usage:
    python -m src.llm.recommendation --darshan-log path/to/log.darshan
    python -m src.llm.recommendation --test-on-benchmark  # run on GT test set
"""

import json
import hashlib
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add local packages path for anthropic/openai
LOCAL_PKGS = Path(__file__).resolve().parent.parent.parent / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

DIM_DESCRIPTIONS = {
    "access_granularity": "I/O operations with very small transfer sizes (<1MB)",
    "metadata_intensity": "Excessive file metadata operations relative to data I/O",
    "parallelism_efficiency": "Uneven I/O load distribution across MPI ranks",
    "access_pattern": "Random (non-sequential) file access pattern",
    "interface_choice": "Using suboptimal I/O interface (e.g., POSIX instead of MPI-IO collective)",
    "file_strategy": "Suboptimal file access strategy (too many files or shared-file contention)",
    "throughput_utilization": "Throughput significantly below achievable (excessive sync, single-OST)",
    "healthy": "No significant I/O bottleneck detected",
}


def load_knowledge_base(kb_path=None):
    """Load the full Knowledge Base."""
    if kb_path is None:
        kb_path = PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json"
    with open(kb_path) as f:
        return json.load(f)


def retrieve_relevant_entries(kb_entries, detected_dims, darshan_signature, top_k=3):
    """Simple retrieval: filter by bottleneck type, rank by feature similarity.

    For production: replace with FAISS vector similarity.
    For now: exact label match + feature distance ranking.
    """
    candidates = []
    for entry in kb_entries:
        # Filter: must share at least one detected bottleneck label
        shared_labels = set(entry["bottleneck_labels"]) & set(detected_dims)
        if not shared_labels:
            continue

        # Score: feature similarity (cosine on key Darshan counters)
        entry_sig = entry.get("darshan_signature", {})
        similarity = 0
        n_common = 0
        for key in darshan_signature:
            if key in entry_sig and darshan_signature[key] != 0:
                ratio = min(darshan_signature[key], entry_sig[key]) / max(darshan_signature[key], entry_sig[key], 1e-9)
                similarity += ratio
                n_common += 1
        if n_common > 0:
            similarity /= n_common

        candidates.append({
            "entry": entry,
            "shared_labels": list(shared_labels),
            "similarity": similarity,
        })

    # Sort by number of shared labels (primary), then similarity (secondary)
    candidates.sort(key=lambda x: (len(x["shared_labels"]), x["similarity"]), reverse=True)
    return candidates[:top_k]


def build_structured_prompt(ml_results, shap_features, kb_entries, darshan_summary):
    """Build the structured LLM prompt with grounded evidence."""

    system_prompt = """You are an HPC I/O performance expert. You analyze Darshan profiling data
and provide specific, actionable code-level optimization recommendations.

RULES:
1. Only recommend fixes that are supported by the benchmark evidence provided.
2. Include specific code snippets showing before/after changes.
3. Cite the benchmark entry ID for each recommendation.
4. Quantify expected improvement using benchmark measurements.
5. If you are unsure, say so explicitly. Do NOT fabricate performance numbers.
6. Prioritize recommendations by expected impact (highest speedup first).
"""

    # Format ML detection results
    detected = []
    for dim in DIMENSIONS:
        conf = ml_results.get(dim, 0)
        if conf > 0.3 and dim != "healthy":
            detected.append(f"  - {dim}: confidence={conf:.2f} — {DIM_DESCRIPTIONS[dim]}")
    if not detected:
        detected.append("  - healthy: No bottlenecks detected")
    detection_str = "\n".join(detected)

    # Format SHAP features
    shap_str = ""
    for dim, features in shap_features.items():
        if features:
            shap_str += f"\n  {dim}:\n"
            for f in features[:5]:
                direction = "high" if f["shap_value"] > 0 else "low"
                shap_str += f"    - {f['feature']} = {f['feature_value']:.4f} (impact: {direction}, |SHAP|={abs(f['shap_value']):.4f})\n"

    # Format KB evidence
    kb_str = ""
    for i, entry in enumerate(kb_entries):
        e = entry["entry"]
        kb_str += f"\n  --- Benchmark Evidence {i+1} (ID: {e['entry_id']}) ---\n"
        kb_str += f"  Benchmark: {e['benchmark']} | Scenario: {e['scenario']}\n"
        kb_str += f"  Bottleneck types: {', '.join(e['bottleneck_labels'])}\n"
        for fix in e.get("fixes", [])[:2]:
            kb_str += f"  Cause: {fix.get('cause', 'N/A')}\n"
            kb_str += f"  Fix: {fix.get('fix', 'N/A')}\n"
            if fix.get("code_before"):
                kb_str += f"  Code before: {fix['code_before']}\n"
            if fix.get("code_after"):
                kb_str += f"  Code after: {fix['code_after']}\n"

    # Format Darshan summary
    summary_str = ""
    for key, val in darshan_summary.items():
        if val and val != 0:
            summary_str += f"  {key}: {val}\n"

    user_prompt = f"""Analyze this HPC job's I/O behavior and provide optimization recommendations.

## Detected Bottlenecks (from ML classifier, 0.923 Micro-F1):
{detection_str}

## Key Contributing Features (SHAP attribution):
{shap_str}

## Benchmark Evidence (from verified Knowledge Base):
{kb_str}

## Job Darshan Summary:
{summary_str}

## Task:
1. Explain what I/O problems this job has, in plain language.
2. For each detected bottleneck, provide a specific code-level fix.
3. Include before/after code snippets.
4. Estimate expected improvement based on benchmark evidence (cite entry IDs).
5. Prioritize by expected impact.

Respond in JSON format:
{{
  "diagnosis": "plain language explanation",
  "recommendations": [
    {{
      "priority": 1,
      "bottleneck": "dimension_name",
      "explanation": "what's wrong",
      "code_before": "problematic code pattern",
      "code_after": "optimized code pattern",
      "expected_speedup": "Nx based on benchmark evidence",
      "kb_citation": "entry_id from KB above",
      "confidence": "high/medium/low"
    }}
  ],
  "overall_expected_improvement": "estimated total speedup"
}}
"""

    return system_prompt, user_prompt


def call_llm(system_prompt, user_prompt, model="claude-sonnet", temperature=0.0,
             max_tokens=2000, cache_dir=None):
    """Call LLM with caching for reproducibility."""

    # Generate cache key from prompt content
    cache_key = hashlib.md5((system_prompt + user_prompt + model).encode()).hexdigest()

    if cache_dir:
        cache_path = Path(cache_dir) / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info("  Cache hit: %s", cache_path.name)
            return cached["response"], cached["metadata"]

    # Call the appropriate API
    t0 = time.perf_counter()

    if "claude" in model.lower():
        import anthropic
        client = anthropic.Anthropic()
        model_id = "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
    elif "gpt" in model.lower():
        from openai import OpenAI
        client = OpenAI()
        model_id = "gpt-4o-2024-11-20"
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
    else:
        raise ValueError(f"Unknown model: {model}")

    latency_ms = (time.perf_counter() - t0) * 1000

    metadata = {
        "model": model,
        "model_id": model_id if 'model_id' in dir() else model,
        "temperature": temperature,
        "latency_ms": round(latency_ms, 1),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Cache the result
    if cache_dir:
        cache_path = Path(cache_dir) / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"response": text, "metadata": metadata,
                        "system_prompt": system_prompt, "user_prompt": user_prompt}, f, indent=2)
        logger.info("  Cached: %s", cache_path.name)

    return text, metadata


def check_groundedness(response_text, kb_entries):
    """Check how many claims in the LLM response are grounded in KB evidence."""
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError:
        return {"parse_error": True, "groundedness_score": 0.0}

    recommendations = response.get("recommendations", [])
    if not recommendations:
        return {"n_recommendations": 0, "groundedness_score": 0.0}

    kb_ids = {e["entry"]["entry_id"] for e in kb_entries}
    grounded = 0
    total = len(recommendations)

    for rec in recommendations:
        citation = rec.get("kb_citation", "")
        if citation in kb_ids:
            grounded += 1

    return {
        "n_recommendations": total,
        "n_grounded": grounded,
        "groundedness_score": grounded / max(total, 1),
    }


def recommend_for_sample(features, labels, models, feature_cols, kb_all,
                          shap_dict=None, sample_idx=0, model="claude-sonnet",
                          cache_dir=None):
    """Generate recommendation for a single sample."""

    # ML predictions
    X = np.array([[features.get(col, 0) for col in feature_cols]], dtype=np.float32)
    ml_results = {}
    for dim in DIMENSIONS:
        if dim in models:
            ml_results[dim] = round(float(models[dim].predict_proba(X)[0][1]), 4)

    # Detected dimensions
    detected_dims = [d for d in DIMENSIONS if ml_results.get(d, 0) > 0.3 and d != "healthy"]
    if not detected_dims:
        detected_dims = ["healthy"]

    # SHAP features
    shap_features = {}
    if shap_dict and sample_idx < len(list(shap_dict.values())[0]):
        for dim in detected_dims:
            if dim in shap_dict:
                sv = shap_dict[dim][sample_idx]
                top_idx = np.argsort(np.abs(sv))[-10:][::-1]
                shap_features[dim] = [
                    {"feature": feature_cols[i], "shap_value": float(sv[i]),
                     "feature_value": float(X[0, i])}
                    for i in top_idx
                ]

    # Darshan signature
    key_features = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                     "avg_write_size", "small_io_ratio", "seq_write_ratio",
                     "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
    darshan_summary = {}
    for f in key_features:
        val = features.get(f, 0)
        if val and not (isinstance(val, float) and np.isnan(val)):
            darshan_summary[f] = round(float(val), 4)

    # Retrieve KB entries
    kb_matches = retrieve_relevant_entries(kb_all, detected_dims, darshan_summary, top_k=3)

    # Build prompt
    system_prompt, user_prompt = build_structured_prompt(
        ml_results, shap_features, kb_matches, darshan_summary
    )

    # Call LLM
    response_text, metadata = call_llm(
        system_prompt, user_prompt, model=model, cache_dir=cache_dir
    )

    # Check groundedness
    groundedness = check_groundedness(response_text, kb_matches)

    return {
        "ml_results": ml_results,
        "detected_dims": detected_dims,
        "shap_features": shap_features,
        "kb_matches": [{"entry_id": m["entry"]["entry_id"],
                         "similarity": m["similarity"],
                         "shared_labels": m["shared_labels"]}
                        for m in kb_matches],
        "response": response_text,
        "metadata": metadata,
        "groundedness": groundedness,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IOSage Single-Shot LLM Recommendation Pipeline")
    parser.add_argument("--model", default="claude-sonnet", choices=["claude-sonnet", "gpt-4o"])
    parser.add_argument("--n-samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--cache-dir", default="data/llm_cache/")
    args = parser.parse_args()

    # Load everything
    logger.info("Loading models and data...")
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        config = yaml.safe_load(f)

    with open(PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl", "rb") as f:
        models = pickle.load(f)

    kb_all = load_knowledge_base()

    # Load SHAP if available
    shap_dict = None
    shap_path = PROJECT_DIR / "paper" / "figures" / "shap" / "shap_values.pkl"
    if shap_path.exists():
        with open(shap_path, "rb") as f:
            shap_data = pickle.load(f)
        shap_dict = shap_data["shap_dict"]

    # Load test data
    prod_feat = pd.read_parquet(PROJECT_DIR / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]

    test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
    test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

    logger.info("Loaded: %d KB entries, %d test samples", len(kb_all), len(test_feat))

    # Run on a few test samples
    results = []
    for i in range(min(args.n_samples, len(test_feat))):
        logger.info("")
        logger.info("Sample %d/%d (benchmark: %s, scenario: %s)...",
                    i + 1, args.n_samples,
                    test_labels.iloc[i].get("benchmark", "?"),
                    test_labels.iloc[i].get("scenario", "?"))

        features = test_feat.iloc[i].to_dict()

        result = recommend_for_sample(
            features, test_labels.iloc[i], models, feature_cols, kb_all,
            shap_dict=shap_dict, sample_idx=i,
            model=args.model, cache_dir=str(PROJECT_DIR / args.cache_dir),
        )

        results.append(result)

        logger.info("  Detected: %s", result["detected_dims"])
        logger.info("  Groundedness: %.2f (%d/%d citations verified)",
                    result["groundedness"]["groundedness_score"],
                    result["groundedness"].get("n_grounded", 0),
                    result["groundedness"].get("n_recommendations", 0))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RECOMMENDATION PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("Samples processed: %d", len(results))
    scores = [r["groundedness"]["groundedness_score"] for r in results
              if not r["groundedness"].get("parse_error")]
    if scores:
        logger.info("Mean groundedness: %.3f", np.mean(scores))
    logger.info("Cache: %s", args.cache_dir)
    logger.info("=" * 60)

    # Save results
    results_path = PROJECT_DIR / "results" / "llm_recommendations.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
