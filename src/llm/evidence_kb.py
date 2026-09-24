"""Build a knowledge base only from measured, job-disjoint fix evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS
from src.ioprescriber.contracts import KB_SCHEMA_VERSION, validate_knowledge_base


MEASUREMENT_SCHEMA_VERSION = 1
SIGNATURE_FEATURES = (
    "nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ",
    "POSIX_WRITES", "POSIX_READS", "POSIX_FSYNCS", "POSIX_OPENS",
    "POSIX_F_META_TIME", "MPIIO_COLL_WRITES", "MPIIO_INDEP_WRITES",
    "avg_write_size", "avg_read_size", "small_io_ratio", "seq_write_ratio",
    "metadata_time_ratio", "collective_ratio", "total_bw_mb_s", "fsync_ratio",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identities(features):
    required = {"_benchmark", "_ground_truth_job_id", "_source_path", "_scenario"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"benchmark features lack identity fields {sorted(missing)}")
    groups = (features["_benchmark"].astype(str) + "/" +
              features["_ground_truth_job_id"].astype(str)).to_numpy()
    ids = (pd.Series(groups, index=features.index) + "/" +
           features["_source_path"].map(lambda value: Path(value).name)).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark sample IDs are not unique")
    return ids, groups


def _load_measurements(path):
    with open(path) as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(f"measurement schema must be {MEASUREMENT_SCHEMA_VERSION}")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("measurement document has no records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("measurement records must be objects")
    ids = [record.get("sample_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every measurement needs a sample_id")
    if len(ids) != len(set(ids)):
        raise ValueError("measurement document has duplicate sample IDs")
    return {record["sample_id"]: record for record in records}


def _verify_measurement_artifacts(records, document_path):
    """Require every result path named by a measurement to exist on disk."""
    base = Path(document_path).resolve().parent
    for sample_id, record in records.items():
        fixes = record.get("fixes")
        if not isinstance(fixes, list):
            raise ValueError(f"measurement {sample_id} has no fix list")
        for fix in fixes:
            measurement = fix.get("measurement", {}) if isinstance(fix, dict) else {}
            paths = list(measurement.get("result_paths", []))
            correctness = measurement.get("correctness_result_path")
            if correctness:
                paths.append(correctness)
            for value in paths:
                path = Path(value)
                resolved = path if path.is_absolute() else base / path
                if not resolved.is_file():
                    raise ValueError(
                        f"measurement {sample_id} artifact does not exist: {value}")


def _load_shap(path, expected_feature_names):
    if path is None:
        return None
    with open(path, "rb") as handle:
        artifact = pickle.load(handle)
    required = {"sample_ids", "feature_names", "shap_dict"}
    if not isinstance(artifact, dict) or not required <= set(artifact):
        raise ValueError("SHAP artifact lacks stable sample IDs or feature names")
    sample_ids = list(map(str, artifact["sample_ids"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("SHAP artifact has duplicate sample IDs")
    if list(artifact["feature_names"]) != list(expected_feature_names):
        raise ValueError("SHAP feature order differs from the model contract")
    for dimension, values in artifact["shap_dict"].items():
        array = np.asarray(values)
        if dimension not in BOTTLENECK_DIMENSIONS:
            raise ValueError(f"SHAP artifact has unknown dimension {dimension}")
        if array.shape != (len(sample_ids), len(expected_feature_names)):
            raise ValueError(f"SHAP array has wrong shape for {dimension}: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"SHAP array has non-finite values for {dimension}")
    return artifact, {sample_id: index for index, sample_id in enumerate(sample_ids)}


def _validate_partitions(partitions, row_count):
    """Require three disjoint partitions that cover each benchmark row once."""
    normalized = []
    for name, values in partitions.items():
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if len(array) != len(np.unique(array)):
            raise ValueError(f"{name} contains duplicate indices")
        if np.any(array < 0) or np.any(array >= row_count):
            raise ValueError(f"{name} contains an out-of-range index")
        normalized.append(array.astype(int))
    combined = np.concatenate(normalized)
    if len(combined) != row_count or not np.array_equal(
            np.sort(combined), np.arange(row_count)):
        raise ValueError("benchmark partitions must cover every row exactly once")
    return normalized


def build_knowledge_base(features_path, labels_path, splits_path, measurements_path,
                         feature_names, shap_path=None):
    """Return a schema 2 KB from accepted training and validation job groups."""
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    if len(features) != len(labels):
        raise ValueError("benchmark features and labels differ in length")
    sample_ids, groups = _identities(features)
    with np.load(splits_path, allow_pickle=False) as split:
        required = {"bench_train", "bench_val", "bench_test", "bench_ids", "bench_groups"}
        if not required <= set(split.files):
            raise ValueError("training splits lack benchmark identities or partitions")
        saved_ids = split["bench_ids"].astype(str)
        saved_groups = split["bench_groups"].astype(str)
        if not np.array_equal(saved_ids, sample_ids) or not np.array_equal(saved_groups, groups):
            raise ValueError("training split identities differ from benchmark inputs")
        train_idx, val_idx, test_idx = _validate_partitions({
            "bench_train": split["bench_train"],
            "bench_val": split["bench_val"],
            "bench_test": split["bench_test"],
        }, len(features))
        allowed_idx = np.concatenate((train_idx, val_idx))
    if len(allowed_idx) == 0 or len(test_idx) == 0:
        raise ValueError("KB construction needs nonempty development and test partitions")
    if set(groups[allowed_idx]) & set(groups[test_idx]):
        raise ValueError("KB and final evaluation partitions share a job group")
    allowed_ids = set(sample_ids[allowed_idx])
    records = _load_measurements(measurements_path)
    _verify_measurement_artifacts(records, measurements_path)
    if not set(records) <= allowed_ids:
        bad = sorted(set(records) - allowed_ids)
        raise ValueError(f"measurement records include final-evaluation samples: {bad[:3]}")

    missing_features = [name for name in feature_names if name not in features.columns]
    if missing_features:
        raise ValueError(f"benchmark features lack model columns {missing_features[:5]}")
    shap = _load_shap(shap_path, feature_names)
    entries = []
    for sample_id, record in records.items():
        position = int(np.flatnonzero(sample_ids == sample_id)[0])
        feature_row = features.iloc[position]
        label_row = labels.iloc[position]
        benchmark = str(feature_row["_benchmark"])
        scenario = str(feature_row["_scenario"])
        if str(label_row.get("benchmark")) != benchmark or str(label_row.get("scenario")) != scenario:
            raise ValueError(f"feature and label identity mismatch for {sample_id}")
        labels_active = [dimension for dimension in BOTTLENECK_DIMENSIONS
                         if int(label_row.get(dimension, 0)) == 1]
        if not labels_active:
            raise ValueError(f"measured fix sample has no bottleneck label: {sample_id}")
        signature = {}
        for name in SIGNATURE_FEATURES:
            value = feature_row.get(name)
            if value is not None and isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                signature[name] = float(value)
        entry = {
            "entry_id": f"kb2:{sample_id}",
            "sample_id": sample_id,
            "job_group": str(groups[position]),
            "benchmark": benchmark,
            "scenario": scenario,
            "bottleneck_labels": labels_active,
            "darshan_signature": signature,
            "source_code": record.get("source_code"),
            "fixes": record.get("fixes"),
            "shap_top_features": {},
        }
        if shap is not None:
            artifact, shap_positions = shap
            if sample_id not in shap_positions:
                raise ValueError(f"SHAP artifact lacks measured sample {sample_id}")
            shap_position = shap_positions[sample_id]
            for dimension in labels_active:
                if dimension not in artifact["shap_dict"]:
                    raise ValueError(f"SHAP artifact lacks dimension {dimension}")
                values = np.asarray(artifact["shap_dict"][dimension])[shap_position]
                top = np.argsort(np.abs(values))[-10:][::-1]
                entry["shap_top_features"][dimension] = [
                    {"feature": feature_names[index], "shap_value": float(values[index]),
                     "feature_value": float(feature_row[feature_names[index]])}
                    for index in top
                ]
        entries.append(entry)

    document = {
        "schema_version": KB_SCHEMA_VERSION,
        "allowed_sample_ids": sorted(allowed_ids),
        "allowed_job_groups": sorted(set(groups[allowed_idx])),
        "entries": entries,
        "inputs": {
            "features_sha256": _sha256(features_path),
            "labels_sha256": _sha256(labels_path),
            "splits_sha256": _sha256(splits_path),
            "measurements_sha256": _sha256(measurements_path),
            "shap_sha256": _sha256(shap_path) if shap_path else None,
        },
    }
    validate_knowledge_base(document)
    return document


def write_knowledge_base(document, output_path):
    """Publish one complete KB without replacing an existing artifact."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"knowledge base already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.incomplete.{os.getpid()}")
    with open(temporary, "x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(description="Build the measured IOSage knowledge base")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--shap")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.bundle, "rb") as handle:
        bundle = pickle.load(handle)
    from src.models.biquality import validate_bundle, verify_bundle_inputs
    validate_bundle(bundle)
    verify_bundle_inputs(bundle)
    if not bundle["final_evaluation"]:
        raise ValueError("KB construction requires a final-evaluation bundle")
    expected = bundle["input_hashes"]
    if _sha256(args.features) != expected["benchmark_features"]["sha256"]:
        raise ValueError("KB feature input differs from the model bundle")
    if _sha256(args.labels) != expected["benchmark_labels"]["sha256"]:
        raise ValueError("KB label input differs from the model bundle")
    document = build_knowledge_base(
        args.features, args.labels, args.splits, args.measurements,
        bundle["feature_names"], args.shap)
    write_knowledge_base(document, args.output)


if __name__ == "__main__":
    main()
