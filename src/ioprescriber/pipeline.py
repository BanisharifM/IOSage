"""
IOPrescriber: End-to-End Pipeline.

Full pipeline: ML detect → SHAP explain → RAG retrieve → LLM recommend → validate speedup

Usage:
    # Analyze a Darshan log (no SLURM, uses cached KB)
    python -m src.ioprescriber.pipeline --darshan-log path/to/file.darshan

    # Run closed-loop on known benchmark pairs (submits SLURM jobs)
    python -m src.ioprescriber.pipeline --closed-loop --submit

    # Dry run (no SLURM, no LLM API)
    python -m src.ioprescriber.pipeline --closed-loop --dry-run

    # Run on GT test set samples
    python -m src.ioprescriber.pipeline --test-samples 5
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

LOCAL_PKGS = Path(__file__).resolve().parent.parent.parent / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class IOPrescriber:
    """Full pipeline: detect → explain → retrieve → recommend → validate."""

    def __init__(self, llm_model="claude-sonnet", cache_dir=None):
        from src.ioprescriber.detector import Detector
        from src.ioprescriber.explainer import Explainer
        from src.ioprescriber.retriever import Retriever
        from src.ioprescriber.recommender import Recommender

        logger.info("Initializing IOPrescriber pipeline...")

        self.detector = Detector()
        self.explainer = Explainer(
            self.detector.models, self.detector.feature_cols, top_k=10
        )
        self.retriever = Retriever()
        self.recommender = Recommender(
            model=llm_model,
            cache_dir=cache_dir or str(PROJECT_DIR / "data" / "llm_cache" / "ioprescriber"),
        )

        logger.info("IOPrescriber ready: detector=%d models, KB=%d entries, LLM=%s",
                    len(self.detector.models), len(self.retriever.kb),
                    self.recommender.model_id)

    def analyze(self, darshan_features, workload_name="unknown"):
        """Run full analysis pipeline on a feature dict.

        Returns complete analysis result with all pipeline outputs.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("IOPrescriber Analysis: %s", workload_name)
        logger.info("=" * 60)
        t0 = time.perf_counter()

        # Step 1: Detect
        logger.info("Step 1: ML Detection...")
        predictions, detected = self.detector.detect_from_features(darshan_features)
        logger.info("  Detected: %s", detected)

        # Step 2: Explain
        logger.info("Step 2: SHAP Attribution...")
        X = np.array([[darshan_features.get(col, 0) for col in self.detector.feature_cols]],
                      dtype=np.float32)
        shap_features = self.explainer.explain(X, detected_dims=detected)
        for dim in detected:
            if dim in shap_features and shap_features[dim]:
                top_feat = shap_features[dim][0]
                logger.info("  %s: top feature = %s (|SHAP|=%.4f)",
                            dim, top_feat["feature"], top_feat["abs_importance"])

        # Step 3: Retrieve
        logger.info("Step 3: KB Retrieval...")
        kb_entries = self.retriever.retrieve(detected, darshan_features)
        logger.info("  Retrieved %d KB entries", len(kb_entries))

        # Step 4: Recommend (if API key available)
        recommendation = None
        groundedness = None
        metadata = None
        raw_response = None

        if self.recommender.api_key:
            logger.info("Step 4: LLM Recommendation...")
            # Build darshan summary
            summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                             "avg_write_size", "small_io_ratio", "seq_write_ratio",
                             "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
            darshan_summary = {k: round(float(darshan_features.get(k, 0)), 4)
                               for k in summary_keys
                               if darshan_features.get(k, 0) != 0}

            recommendation, groundedness, metadata, raw_response = self.recommender.recommend(
                predictions, detected, shap_features, kb_entries, darshan_summary
            )

            if recommendation:
                n_recs = len(recommendation.get("recommendations", []))
                logger.info("  Generated %d recommendations", n_recs)
                logger.info("  Groundedness: %.2f",
                            groundedness.get("groundedness_score", 0))
            else:
                logger.warning("  LLM response could not be parsed")
        else:
            logger.info("Step 4: SKIPPED (no API key set)")

        total_ms = (time.perf_counter() - t0) * 1000

        result = {
            "workload": workload_name,
            "pipeline_latency_ms": round(total_ms, 1),
            "step1_detection": {
                "predictions": predictions,
                "detected": detected,
            },
            "step2_shap": {dim: feats[:3] for dim, feats in shap_features.items()},
            "step3_retrieval": {
                "n_entries": len(kb_entries),
                "entries": [{"entry_id": e["entry"]["entry_id"],
                             "similarity": e["similarity"],
                             "matched_dims": e["matched_dims"]}
                            for e in kb_entries],
            },
            "step4_recommendation": {
                "parsed": recommendation,
                "groundedness": groundedness,
                "metadata": metadata,
            },
        }

        logger.info("")
        logger.info("Pipeline completed in %.0fms", total_ms)
        logger.info("=" * 60)

        return result

    def analyze_darshan_log(self, darshan_path):
        """Analyze directly from a Darshan log file."""
        predictions, detected, features = self.detector.detect_from_darshan(darshan_path)
        return self.analyze(features, workload_name=Path(darshan_path).stem)


def main():
    parser = argparse.ArgumentParser(description="IOPrescriber: ML+LLM I/O Bottleneck Diagnosis")
    parser.add_argument("--darshan-log", help="Path to Darshan log file")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="Run on N benchmark test samples")
    parser.add_argument("--closed-loop", action="store_true",
                        help="Run closed-loop validation on known pairs")
    parser.add_argument("--submit", action="store_true",
                        help="Submit SLURM jobs (requires --closed-loop)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call LLM API or submit SLURM")
    parser.add_argument("--llm-model", default="claude-sonnet",
                        choices=["claude-sonnet", "gpt-4o", "llama-70b"])
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

    if args.dry_run:
        os.environ.pop("OPENROUTER_API_KEY", None)

    pipeline = IOPrescriber(llm_model=args.llm_model)

    results = []

    if args.darshan_log:
        result = pipeline.analyze_darshan_log(args.darshan_log)
        results.append(result)

    elif args.test_samples > 0:
        test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
        test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

        # Pick samples with bottlenecks (not healthy)
        bottleneck_mask = test_labels["healthy"] == 0
        indices = test_feat.index[bottleneck_mask.values].tolist()
        n = min(args.test_samples, len(indices))

        for i in range(n):
            idx = indices[i]
            features = test_feat.iloc[idx].to_dict()
            name = f"{test_labels.iloc[idx].get('benchmark', '?')}_{test_labels.iloc[idx].get('scenario', '?')}"
            result = pipeline.analyze(features, workload_name=name)
            results.append(result)

    elif args.closed_loop:
        from src.ioprescriber.validator import Validator
        validator = Validator()
        cl_results = validator.validate_all(submit_jobs=args.submit)
        results.append({"closed_loop": cl_results})

    else:
        parser.print_help()
        return

    # Save results
    results_dir = PROJECT_DIR / "results" / "ioprescriber"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"pipeline_results_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
