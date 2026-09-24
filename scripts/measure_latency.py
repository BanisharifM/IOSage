#!/usr/bin/env python3
"""Measure repeated complete pipeline requests in live or cached mode."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ioprescriber.contracts import validate_pipeline_result
from src.ioprescriber.pipeline import IOPrescriber


def run_request(pipeline, features, workload, namespace, bypass_cache):
    started = time.perf_counter()
    result = pipeline.analyze(
        features,
        workload_name=workload,
        sample_id=f"latency/{namespace}",
        job_group="latency/study",
        cache_namespace=namespace,
        bypass_cache=bypass_cache,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    validate_pipeline_result(result)
    recommendation = result["recommendation"]
    if recommendation["parsed"] is None:
        raise ValueError("latency request did not produce a parsed recommendation")
    return elapsed_ms, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--run-id", required=True,
                        help="Unique identifier for this measurement run")
    parser.add_argument("--mode", required=True, choices=("live", "cached"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    feature_path = Path(args.features_json).resolve()
    features = json.loads(feature_path.read_text())
    if not isinstance(features, dict) or not features:
        raise ValueError("features JSON must be a nonempty object")

    pipeline = IOPrescriber(args.model_bundle, args.knowledge_base,
                            llm_model="claude-sonnet", use_shap=True)
    warmup_ms = None
    namespace = f"latency:{args.run_id}:{args.mode}:{feature_path.name}"
    if args.mode == "cached":
        warmup_ms, warmup = run_request(
            pipeline, features, args.workload, namespace, bypass_cache=False)
        if warmup["recommendation"]["metadata"].get("cache_hit") is True:
            raise ValueError("cached-mode warmup unexpectedly used an existing entry")

    samples = []
    request_metadata = []
    for repeat in range(args.repeats):
        request_namespace = namespace if args.mode == "cached" else f"{namespace}:{repeat}"
        elapsed_ms, result = run_request(
            pipeline, features, args.workload, request_namespace,
            bypass_cache=args.mode == "live")
        metadata = result["recommendation"]["metadata"]
        expected_hit = args.mode == "cached"
        if metadata.get("cache_hit") is not expected_hit:
            raise ValueError(
                f"request {repeat} cache mode differs from declared {args.mode} mode")
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError("latency sample must be positive and finite")
        samples.append(elapsed_ms)
        request_metadata.append(metadata)

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "run_id": args.run_id,
        "workload": args.workload,
        "features": str(feature_path),
        "repeats": args.repeats,
        "warmup_ms": warmup_ms,
        "request_latency_ms": samples,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": statistics.mean(samples),
        "request_metadata": request_metadata,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
