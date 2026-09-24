"""
IOPrescriber: End-to-End Pipeline.

Full pipeline: ML detection, SHAP attribution, evidence retrieval, and LLM recommendation.

Use ``python -m src.ioprescriber.pipeline --help`` for the required model and
measured-evidence inputs. The fixed-pair validation command is retired.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class IOPrescriber:
    """Full detection, attribution, retrieval, and recommendation pipeline."""

    def __init__(self, model_path, kb_path, llm_model="claude-sonnet",
                 cache_dir=None, use_shap=True):
        """``model_path``: a model bundle from ``scripts/train_biquality.py``."""
        from src.ioprescriber.detector import Detector
        from src.ioprescriber.retriever import Retriever
        from src.ioprescriber.recommender import Recommender

        logger.info("Initializing IOPrescriber pipeline...")

        self.use_shap = use_shap
        self.detector = Detector(model_path)
        self.explainer = None
        if self.use_shap:
            from src.ioprescriber.explainer import Explainer
            self.explainer = Explainer(
                self.detector.models, self.detector.feature_cols, top_k=10
            )
            logger.info("SHAP explainer loaded")
        self.retriever = Retriever(kb_path=kb_path)
        self.recommender = Recommender(
            model=llm_model,
            cache_dir=cache_dir or str(PROJECT_DIR / "data" / "llm_cache" / "ioprescriber"),
        )

        logger.info("IOPrescriber ready: detector=%d models, KB=%d entries, LLM=%s, SHAP=%s",
                    len(self.detector.models), len(self.retriever.kb),
                    self.recommender.model_id, self.use_shap)

    def analyze(self, darshan_features, workload_name="unknown", sample_id=None,
                job_group=None, cache_namespace=None, bypass_cache=False):
        """Run full analysis pipeline on a feature dict.

        Returns complete analysis result with all pipeline outputs.
        """
        if sample_id is None and {
                "_benchmark", "_ground_truth_job_id", "_source_path"} <= set(darshan_features):
            job_group = (f"{darshan_features['_benchmark']}/"
                         f"{darshan_features['_ground_truth_job_id']}")
            sample_id = f"{job_group}/{Path(str(darshan_features['_source_path'])).name}"

        logger.info("")
        logger.info("=" * 60)
        logger.info("IOPrescriber Analysis: %s", workload_name)
        logger.info("=" * 60)
        t0 = time.perf_counter()

        # Step 1: Detect
        logger.info("Step 1: ML Detection...")
        predictions, detected = self.detector.detect_from_features(darshan_features)
        logger.info("  Detected: %s", detected)

        # Step 2: Explain (optional, offline analysis only)
        shap_features = {}
        if self.use_shap and self.explainer:
            logger.info("Step 2: SHAP Attribution (optional)...")
            X = self.detector.feature_vector(darshan_features)
            shap_features = self.explainer.explain(X, detected_dims=detected)
            for dim in detected:
                if dim in shap_features and shap_features[dim]:
                    top_feat = shap_features[dim][0]
                    logger.info("  %s: top feature = %s (|SHAP|=%.4f)",
                                dim, top_feat["feature"], top_feat["abs_importance"])

        # Step 2: Retrieve
        logger.info("Step 2: KB Retrieval...")
        kb_entries = self.retriever.retrieve(
            detected, darshan_features, query_sample_id=sample_id,
            query_job_group=job_group)
        logger.info("  Retrieved %d KB entries", len(kb_entries))

        # Step 3: Recommend (if API key available)
        recommendation = None
        groundedness = None
        metadata = None

        needs_recommendation = detected != ["healthy"]
        if needs_recommendation and not kb_entries:
            raise ValueError("no measured KB evidence matches the detected bottlenecks")
        if self.recommender.api_key and needs_recommendation:
            logger.info("Step 3: LLM Recommendation...")
            # Build darshan summary
            summary_keys = ["nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN",
                             "avg_write_size", "small_io_ratio", "seq_write_ratio",
                             "metadata_time_ratio", "collective_ratio", "total_bw_mb_s"]
            darshan_summary = {k: round(float(darshan_features.get(k, 0)), 4)
                               for k in summary_keys
                               if darshan_features.get(k, 0) != 0}

            recommendation, groundedness, metadata, _ = self.recommender.recommend(
                predictions, detected, shap_features, kb_entries, darshan_summary,
                cache_namespace=cache_namespace, bypass_cache=bypass_cache,
            )

            if recommendation:
                n_recs = len(recommendation.get("recommendations", []))
                logger.info("  Generated %d recommendations", n_recs)
                logger.info("  Groundedness: %.2f",
                            groundedness.get("groundedness_score", 0))
            else:
                logger.warning("  LLM response could not be parsed")
        elif not needs_recommendation:
            logger.info("Step 3: SKIPPED (healthy detection)")
        else:
            logger.info("Step 3: SKIPPED (no API key set)")

        total_ms = (time.perf_counter() - t0) * 1000

        result = {
            "schema_version": 1,
            "workload": workload_name,
            "pipeline_latency_ms": round(total_ms, 1),
            "detection": {
                "predictions": predictions,
                "detected": detected,
            },
            "attribution": {dim: feats[:3] for dim, feats in shap_features.items()} if shap_features else {},
            "retrieval": {
                "n_entries": len(kb_entries),
                "entries": [{"entry_id": e["entry"]["entry_id"],
                             "similarity": e["similarity"],
                             "matched_dims": e["matched_dims"]}
                            for e in kb_entries],
            },
            "recommendation": {
                "parsed": recommendation,
                "groundedness": groundedness,
                "metadata": metadata,
            },
        }

        logger.info("")
        logger.info("Pipeline completed in %.0fms", total_ms)
        logger.info("=" * 60)

        from src.ioprescriber.contracts import validate_pipeline_result
        return validate_pipeline_result(result)

    def analyze_darshan_log(self, darshan_path):
        """Analyze directly from a Darshan log file."""
        predictions, detected, features = self.detector.detect_from_darshan(darshan_path)
        return self.analyze(features, workload_name=Path(darshan_path).stem)


def main():
    parser = argparse.ArgumentParser(description="IOPrescriber: ML+LLM I/O Bottleneck Diagnosis")
    parser.add_argument("--model-bundle", required=True,
                        help="Final-evaluation bundle from train_biquality.py")
    parser.add_argument("--knowledge-base", required=True,
                        help="Schema 2 measured-evidence knowledge base")
    parser.add_argument("--darshan-log", help="Path to Darshan log file")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="Run on N benchmark test samples")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--closed-loop", action="store_true",
                        help="Run closed-loop validation on known pairs")
    parser.add_argument("--submit", action="store_true",
                        help="Submit SLURM jobs (requires --closed-loop)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call LLM API or submit SLURM")
    parser.add_argument("--no-shap", action="store_true",
                        help="Explicit ablation: omit SHAP attribution")
    parser.add_argument("--llm-model", default="claude-sonnet",
                        choices=["claude-sonnet", "gpt-4o", "llama-70b"])
    args = parser.parse_args()
    if args.closed_loop:
        parser.error("the fixed-pair validator is retired; use the source-aware validation path once configured")
    if args.submit and not args.closed_loop:
        parser.error("--submit requires --closed-loop")
    if args.dry_run and args.submit:
        parser.error("--dry-run cannot be combined with --submit")

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

    pipeline = IOPrescriber(
        model_path=args.model_bundle,
        kb_path=args.knowledge_base,
        llm_model=args.llm_model,
        use_shap=not args.no_shap,
    )

    results = []

    if args.darshan_log:
        result = pipeline.analyze_darshan_log(args.darshan_log)
        results.append(result)

    elif args.test_samples > 0:
        from src.models.biquality import load_final_benchmark_test_frames
        _, test_feat, test_labels, _ = load_final_benchmark_test_frames(
            args.model_bundle)

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

    else:
        parser.print_help()
        return

    # Save results
    results_dir = Path(args.output_dir).resolve()
    if results_dir.exists():
        raise FileExistsError(f"pipeline output directory already exists: {results_dir}")
    results_dir.mkdir(parents=True)
    results_path = results_dir / "pipeline_results.json"
    with results_path.open("x") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved: %s", results_path)


if __name__ == "__main__":
    main()
