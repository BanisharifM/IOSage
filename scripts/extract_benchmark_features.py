#!/usr/bin/env python3
"""
Extract features and labels from benchmark Darshan logs.

Produces ground-truth feature vectors and labels with the production
pipeline's columns (``extract_raw_features`` then ``stage3_engineer``).

Samples come from ``src.data.benchmark_logs.iter_benchmark_samples``: one log
for the compiled MPI benchmarks (IOR, mdtest, h5bench, HACC-IO), the merged
per-process logs of one job for DLIO and the custom mpi4py runs. Labels come
from the manifest (``scripts/build_label_manifest.py``); a sample without a
manifest row is an error, and a sample whose row says ``source=none`` is
excluded. A verification report must contain one passing row for every
labeled sample, and its sidecar must name the manifest and the label
definitions in use, before either output is written.

Output:
    <output-dir>/features.parquet   same columns as production/features.parquet
    <output-dir>/labels.parquet     labels, per-target validity, and metadata

Exit status: 0 when every labeled sample was extracted, 1 on a verification,
manifest, parsing, or pipeline error.

Usage:
    python scripts/extract_benchmark_features.py --output-dir data/processed/resubmission/benchmark
    python scripts/extract_benchmark_features.py --bench-type ior
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.data.benchmark_logs import (  # noqa: E402
    AGGREGATED_BENCHMARKS, DEFAULT_MANIFEST, EXCLUDED_SOURCE, PER_RANK_BENCHMARKS,
    iter_benchmark_samples, load_manifest, manifest_row, sidecar_path,
    validate_verification_report)
from src.data.label_rules import DIMENSION_NAMES  # noqa: E402
from src.data.feature_extraction import (  # noqa: E402
    FEATURE_SCHEMA_VERSION, extract_raw_features, get_info_columns)
from src.data.preprocessing import load_preprocessing_config, stage3_engineer  # noqa: E402
from src.utils.artifacts import sha256_file, write_atomic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = PROJECT_DIR / "data" / "benchmark_logs"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "resubmission" / "benchmark"
BENCHMARKS = sorted(AGGREGATED_BENCHMARKS | PER_RANK_BENCHMARKS)
EXTRA_COLUMNS = ["_source_path", "_benchmark", "_scenario", "_ground_truth_job_id"]
LABEL_META = ["job_id", "benchmark", "scenario", "n_ranks", "n_darshan_files", "label_source"]


def extract_benchmark(bench_type, log_dir, manifest):
    """Feature and label rows of one benchmark; returns (features, labels, counts)."""
    feature_rows, label_rows = [], []
    counts = {"extracted": 0, "unlabeled": 0, "unparsed": 0}
    for job_id, files, parsed, error in iter_benchmark_samples(bench_type, str(log_dir)):
        row = manifest_row(manifest, bench_type, job_id, files)
        if row["source"] == EXCLUDED_SOURCE:
            counts["unlabeled"] += 1
            continue
        if parsed is None:
            logger.error("  %s job %s was not parsed: %s", bench_type, job_id, error)
            counts["unparsed"] += 1
            continue
        features = extract_raw_features(parsed)
        features.update(_source_path=files[0], _benchmark=bench_type,
                        _scenario=row["scenario"], _ground_truth_job_id=job_id)
        feature_rows.append(features)
        labels = {"job_id": job_id, "benchmark": bench_type, "scenario": row["scenario"],
                  "n_ranks": features["nprocs"], "n_darshan_files": len(files),
                  "label_source": row["source"]}
        labels.update({d: int(row[d]) for d in DIMENSION_NAMES})
        labels.update({f"valid_{d}": int(row[f"valid_{d}"]) for d in DIMENSION_NAMES})
        label_rows.append(labels)
        counts["extracted"] += 1
    logger.info("  %s: %d extracted, %d unlabeled (left out), %d unparsed (left out)",
                bench_type, counts["extracted"], counts["unlabeled"], counts["unparsed"])
    return feature_rows, label_rows, counts


def main():
    parser = argparse.ArgumentParser(description="Extract features and labels from benchmark Darshan logs")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                        help="Root directory with benchmark_logs/<benchmark>/")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--verification-report", required=True,
                        help="CSV produced by verify_all_ground_truth.py")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "preprocessing.yaml"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bench-type", choices=["all"] + BENCHMARKS, default="all")
    args = parser.parse_args()

    config = load_preprocessing_config(args.config)
    manifest = load_manifest(args.manifest)
    bench_types = BENCHMARKS if args.bench_type == "all" else [args.bench_type]
    verification = validate_verification_report(
        manifest, args.verification_report, args.manifest, bench_types=bench_types)
    logger.info("Verification gate: %s", verification)

    all_features, all_labels = [], []
    unparsed = 0
    for bench in bench_types:
        log_dir = Path(args.log_dir) / bench
        if not log_dir.is_dir():
            raise FileNotFoundError(f"log directory not found: {log_dir}")
        feats, labs, counts = extract_benchmark(bench, log_dir, manifest)
        all_features.extend(feats)
        all_labels.extend(labs)
        unparsed += counts["unparsed"]
    if unparsed:
        raise RuntimeError(f"{unparsed} labeled benchmark samples could not be parsed")
    if not all_features:
        raise RuntimeError("no sample extracted")

    # Same derived features as the production pipeline; the extra columns
    # are carried through untouched (stage 3 leaves _-prefixed columns alone)
    features_df = stage3_engineer(pd.DataFrame(all_features), config=config)
    info_cols = get_info_columns()
    feature_cols = [c for c in features_df.columns if c not in info_cols and c not in EXTRA_COLUMNS]
    features_df = features_df[feature_cols + info_cols + EXTRA_COLUMNS]
    validity_columns = [f"valid_{dimension}" for dimension in DIMENSION_NAMES]
    labels_df = pd.DataFrame(all_labels)[LABEL_META + DIMENSION_NAMES + validity_columns]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = output_dir / "features.parquet"
    label_path = output_dir / "labels.parquet"
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    existing = [str(path) for path in (feat_path, label_path, dataset_manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing outputs: {existing}")
    write_atomic(feat_path, lambda path: features_df.to_parquet(path, index=False))
    write_atomic(label_path, lambda path: labels_df.to_parquet(path, index=False))

    dataset_manifest = {
        "schema_version": 1,
        "status": "passed",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "benchmarks": bench_types,
        "verification": verification,
        "inputs": {
            "preprocessing_config": {
                "path": str(Path(args.config).resolve()),
                "sha256": sha256_file(args.config),
            },
            "label_manifest": {
                "path": str(Path(args.manifest).resolve()),
                "sha256": sha256_file(args.manifest),
            },
            "verification_report": {
                "path": str(Path(args.verification_report).resolve()),
                "sha256": sha256_file(args.verification_report),
            },
            "verification_sidecar": {
                "path": str(sidecar_path(args.verification_report).resolve()),
                "sha256": sha256_file(sidecar_path(args.verification_report)),
            },
        },
        "outputs": {
            "features": {"path": str(feat_path), "sha256": sha256_file(feat_path),
                         "rows": len(features_df), "columns": len(features_df.columns)},
            "labels": {"path": str(label_path), "sha256": sha256_file(label_path),
                       "rows": len(labels_df), "columns": len(labels_df.columns)},
        },
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256_file(Path(__file__).resolve())},
    }
    def write_manifest(path):
        with path.open("x") as handle:
            json.dump(dataset_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    write_atomic(dataset_manifest_path, write_manifest)
    logger.info("Features: %s, labels: %s", features_df.shape, labels_df.shape)
    logger.info("Label positives: %s", labels_df[DIMENSION_NAMES].sum().to_dict())
    logger.info("Saved: %s, %s, and %s", feat_path, label_path, dataset_manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
