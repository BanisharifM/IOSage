"""Shared observable rules for heuristic labels and benchmark checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DIMENSION_NAMES = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "request_alignment",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
    "healthy",
]
BOTTLENECK_DIMENSIONS = DIMENSION_NAMES[:8]

LABEL_DEFINITIONS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "label_definitions.yaml"
)
_THRESHOLD_NAMES = {
    "small_request_max_bytes",
    "small_request_count",
    "small_request_share",
    "metadata_time_share",
    "rank_imbalance_range_ratio",
    "shared_straggler_share",
    "random_request_count",
    "random_request_share",
    "file_misalignment_share",
    "collective_min_operations",
    "many_small_files_count",
    "small_file_mean_bytes",
    "sync_write_count",
    "syncs_per_write",
}
_DEFINITION_FIELDS = {
    "definition", "criterion", "impact", "canonical_fix",
    "when_not_to_apply", "sources",
}


def _load_label_definitions(path: Path = LABEL_DEFINITIONS_PATH) -> dict[str, object]:
    """Load and validate the registered label contract."""
    with path.open() as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version", "thresholds", "dimensions",
    }:
        raise ValueError(f"invalid label-definition structure: {path}")
    if contract["schema_version"] != 2:
        raise ValueError(f"unsupported label-definition schema: {path}")

    thresholds = contract["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != _THRESHOLD_NAMES:
        raise ValueError(f"label-definition thresholds differ from the code contract: {path}")
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"label threshold {name} must be a positive number")
    for name in (
        "small_request_share", "metadata_time_share", "rank_imbalance_range_ratio",
        "shared_straggler_share", "random_request_share", "file_misalignment_share",
        "syncs_per_write",
    ):
        if thresholds[name] > 1:
            raise ValueError(f"label threshold {name} must not exceed 1")
    for name in (
        "small_request_max_bytes", "small_request_count", "random_request_count",
        "collective_min_operations", "many_small_files_count", "small_file_mean_bytes",
        "sync_write_count",
    ):
        if not isinstance(thresholds[name], int):
            raise ValueError(f"label threshold {name} must be an integer")
    if thresholds["small_request_max_bytes"] != 1024 * 1024:
        raise ValueError("small_request_max_bytes must match Darshan's 1 MiB histogram boundary")

    dimensions = contract["dimensions"]
    if not isinstance(dimensions, dict) or list(dimensions) != DIMENSION_NAMES:
        raise ValueError(f"label definitions must list the registered dimensions in order: {path}")
    for dimension, definition in dimensions.items():
        if not isinstance(definition, dict) or set(definition) != _DEFINITION_FIELDS:
            raise ValueError(f"definition fields are incomplete for {dimension}")
        for field in _DEFINITION_FIELDS - {"sources"}:
            if not isinstance(definition[field], str) or not definition[field].strip():
                raise ValueError(f"{dimension}.{field} must be a nonempty string")
        sources = definition["sources"]
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) or not source.strip() for source in sources)):
            raise ValueError(f"{dimension}.sources must be a nonempty string list")
    return contract


LABEL_DEFINITIONS = _load_label_definitions()
_THRESHOLDS = LABEL_DEFINITIONS["thresholds"]
SMALL_REQUEST_MAX_BYTES = int(_THRESHOLDS["small_request_max_bytes"])
SMALL_REQUEST_SHARE = float(_THRESHOLDS["small_request_share"])
SMALL_REQUEST_COUNT = int(_THRESHOLDS["small_request_count"])
RANDOM_REQUEST_SHARE = float(_THRESHOLDS["random_request_share"])
RANDOM_REQUEST_COUNT = int(_THRESHOLDS["random_request_count"])
COLLECTIVE_MIN_OPS = int(_THRESHOLDS["collective_min_operations"])
METADATA_TIME_SHARE = float(_THRESHOLDS["metadata_time_share"])
FSYNC_PER_WRITE = float(_THRESHOLDS["syncs_per_write"])
SYNC_WRITE_COUNT = int(_THRESHOLDS["sync_write_count"])
RANK_IMBALANCE_RANGE_RATIO = float(_THRESHOLDS["rank_imbalance_range_ratio"])
SHARED_STRAGGLER_SHARE = float(_THRESHOLDS["shared_straggler_share"])
FILE_MISALIGNMENT_SHARE = float(_THRESHOLDS["file_misalignment_share"])
MANY_SMALL_FILES_COUNT = int(_THRESHOLDS["many_small_files_count"])
SMALL_FILE_MEAN_BYTES = int(_THRESHOLDS["small_file_mean_bytes"])

_POSIX_SMALL = {
    direction: [
        f"POSIX_SIZE_{direction}_0_100",
        f"POSIX_SIZE_{direction}_100_1K",
        f"POSIX_SIZE_{direction}_1K_10K",
        f"POSIX_SIZE_{direction}_10K_100K",
        f"POSIX_SIZE_{direction}_100K_1M",
    ]
    for direction in ("READ", "WRITE")
}
_MPIIO_SMALL = {
    direction: [name.replace("POSIX_SIZE_", "MPIIO_SIZE_").replace(
        f"{direction}_", f"{direction}_AGG_", 1
    ) for name in names]
    for direction, names in _POSIX_SMALL.items()
}


def _as_frame(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features
    return pd.DataFrame([dict(features)])


def parallelism_present(features: pd.DataFrame | Mapping[str, object]):
    """Evaluate the rank-distribution rule without requiring other features."""
    frame = _as_frame(features)
    result = (
        (frame["nprocs"] > 1)
        & (
            (frame.get("rank_byte_range_ratio", 0.0) > RANK_IMBALANCE_RANGE_RATIO)
            | (frame.get("SHARED_BYTE_IMBALANCE", 0.0) > SHARED_STRAGGLER_SHARE)
            | (frame.get("SHARED_TIME_IMBALANCE", 0.0) > SHARED_STRAGGLER_SHARE)
            | (frame.get("FILE_WRITE_IMBALANCE", 0.0) > RANK_IMBALANCE_RANGE_RATIO)
            | (frame.get("FILE_READ_IMBALANCE", 0.0) > RANK_IMBALANCE_RANGE_RATIO)
        )
    )
    if isinstance(features, pd.DataFrame):
        return result
    return bool(result.iloc[0])


def rule_frame(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Evaluate the registered problem rules on one or more feature rows."""
    frame = _as_frame(features)
    small_direction = []
    for direction in ("READ", "WRITE"):
        posix_ops = frame[f"POSIX_{direction}S"]
        mpiio_direction_ops = (
            frame[f"MPIIO_INDEP_{direction}S"]
            + frame[f"MPIIO_COLL_{direction}S"]
            + frame[f"MPIIO_NB_{direction}S"]
        )
        posix_small = sum(frame[name] for name in _POSIX_SMALL[direction])
        mpiio_small = sum(frame[name] for name in _MPIIO_SMALL[direction])
        use_mpiio_direction = mpiio_direction_ops > 0
        operation_count = np.where(use_mpiio_direction, mpiio_direction_ops, posix_ops)
        small_count = np.where(use_mpiio_direction, mpiio_small, posix_small)
        small_direction.append(
            (small_count > SMALL_REQUEST_COUNT)
            & (small_count / np.maximum(operation_count, 1) > SMALL_REQUEST_SHARE)
        )

    random_direction = []
    for direction in ("READ", "WRITE"):
        operations = frame[f"POSIX_{direction}S"]
        random_operations = np.maximum(
            operations - frame[f"POSIX_SEQ_{direction}S"],
            0,
        )
        random_direction.append(
            (random_operations > RANDOM_REQUEST_COUNT)
            & (random_operations / np.maximum(operations, 1) > RANDOM_REQUEST_SHARE)
        )

    missing_collective = []
    for direction in ("READ", "WRITE"):
        operations = (
            frame[f"MPIIO_INDEP_{direction}S"]
            + frame[f"MPIIO_COLL_{direction}S"]
        )
        missing_collective.append(
            (operations > COLLECTIVE_MIN_OPS)
            & (frame[f"MPIIO_COLL_{direction}S"] == 0)
        )
    writes = frame["POSIX_WRITES"]
    syncs = frame["POSIX_FSYNCS"] + frame["POSIX_FDSYNCS"]
    posix_operations = frame["POSIX_READS"] + writes
    mean_data_file_bytes = frame["io_bytes_all"] / np.maximum(frame["num_data_files"], 1)

    rules = pd.DataFrame(index=frame.index)
    rules["access_granularity"] = small_direction[0] | small_direction[1]
    rules["metadata_intensity"] = (
        frame["metadata_time_ratio_all"] >= METADATA_TIME_SHARE
    )
    rules["parallelism_efficiency"] = parallelism_present(frame)
    rules["access_pattern"] = random_direction[0] | random_direction[1]
    rules["request_alignment"] = (
        (posix_operations > 0)
        & (frame["POSIX_FILE_NOT_ALIGNED"] / np.maximum(posix_operations, 1)
           > FILE_MISALIGNMENT_SHARE)
    )
    rules["interface_choice"] = missing_collective[0] | missing_collective[1]
    rules["file_strategy"] = (
        (frame["num_data_files"] > MANY_SMALL_FILES_COUNT)
        & (mean_data_file_bytes <= SMALL_FILE_MEAN_BYTES)
    )
    rules["throughput_utilization"] = (
        (writes > SYNC_WRITE_COUNT)
        & (syncs / np.maximum(writes, 1) >= FSYNC_PER_WRITE)
    )
    return rules.astype(bool)


def labels_from_features(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Return the problem labels and their derived healthy complement.

    A problem label is its rule decision masked by the target's validity
    (``validity_from_features``): a rule evaluated on an incomplete module
    record set is not evidence, so it never becomes a positive. Healthy is
    the complement of the masked problem labels; whether healthy itself is a
    valid target is ``valid_healthy`` of the same validity frame.
    """
    rules = rule_frame(features)
    validity = validity_from_features(features)[BOTTLENECK_DIMENSIONS].astype(bool)
    labels = (rules & validity).astype(int)
    labels["healthy"] = (~(rules & validity).any(axis=1)).astype(int)
    return labels[DIMENSION_NAMES]


def validity_from_features(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Return target validity for rows with incomplete module records."""
    frame = _as_frame(features)
    posix = frame["partial_posix"].astype(bool)
    mpiio = frame["partial_mpiio"].astype(bool)
    stdio = frame["partial_stdio"].astype(bool)
    validity = pd.DataFrame(index=frame.index)
    validity["access_granularity"] = ~(posix | mpiio)
    validity["metadata_intensity"] = ~(posix | stdio)
    validity["parallelism_efficiency"] = ~(posix | mpiio | stdio)
    validity["access_pattern"] = ~posix
    validity["request_alignment"] = ~posix
    validity["interface_choice"] = ~mpiio
    validity["file_strategy"] = ~(posix | stdio)
    validity["throughput_utilization"] = ~posix
    validity["healthy"] = validity[BOTTLENECK_DIMENSIONS].all(axis=1)
    return validity[DIMENSION_NAMES].astype(int)


def rule_details(features: Mapping[str, object]) -> dict[str, str]:
    """Human-readable values for one row, keyed by bottleneck dimension."""
    f = features
    small_parts = []
    for direction in ("READ", "WRITE"):
        mpiio_ops = sum(float(f[f"MPIIO_{kind}_{direction}S"])
                        for kind in ("INDEP", "COLL", "NB"))
        layer = "mpiio" if mpiio_ops > 0 else "posix"
        if layer == "mpiio":
            operations = mpiio_ops
            small = sum(float(f[name]) for name in _MPIIO_SMALL[direction])
        else:
            operations = float(f[f"POSIX_{direction}S"])
            small = sum(float(f[name]) for name in _POSIX_SMALL[direction])
        small_parts.append(
            f"{direction.lower()}_{layer}_small={small:.0f}/{operations:.0f}"
        )
    posix_reads = float(f["POSIX_READS"])
    posix_writes = float(f["POSIX_WRITES"])
    posix_ops = posix_reads + posix_writes
    random_reads = max(
        posix_reads - float(f["POSIX_SEQ_READS"]), 0)
    random_writes = max(
        posix_writes - float(f["POSIX_SEQ_WRITES"]), 0)
    writes = float(f["POSIX_WRITES"])
    syncs = float(f["POSIX_FSYNCS"]) + float(f["POSIX_FDSYNCS"])
    return {
        "access_granularity": " ".join(small_parts),
        "metadata_intensity": (
            f"metadata_time_ratio_all={float(f['metadata_time_ratio_all']):.3f} "
            f"bytes={float(f['io_bytes_all']):.0f}"
        ),
        "parallelism_efficiency": (
            f"range_ratio={float(f['rank_byte_range_ratio']):.3f} "
            f"shared_byte={float(f['SHARED_BYTE_IMBALANCE']):.3f} "
            f"shared_time={float(f['SHARED_TIME_IMBALANCE']):.3f} "
            f"file_write={float(f['FILE_WRITE_IMBALANCE']):.3f} "
            f"file_read={float(f['FILE_READ_IMBALANCE']):.3f} "
            f"nprocs={float(f['nprocs']):.0f}"
        ),
        "access_pattern": (
            f"random_read={random_reads:.0f}/{posix_reads:.0f} "
            f"random_write={random_writes:.0f}/{posix_writes:.0f}"
        ),
        "request_alignment": (
            f"file_not_aligned={float(f['POSIX_FILE_NOT_ALIGNED']):.0f}/{posix_ops:.0f}"
        ),
        "interface_choice": (
            f"read_indep={float(f['MPIIO_INDEP_READS']):.0f} "
            f"read_coll={float(f['MPIIO_COLL_READS']):.0f} "
            f"write_indep={float(f['MPIIO_INDEP_WRITES']):.0f} "
            f"write_coll={float(f['MPIIO_COLL_WRITES']):.0f}"
        ),
        "file_strategy": (
            f"data_files={float(f['num_data_files']):.0f} "
            f"mean_bytes_per_data_file="
            f"{float(f['io_bytes_all']) / max(float(f['num_data_files']), 1):.0f}"
        ),
        "throughput_utilization": (
            f"syncs_per_write={syncs / max(writes, 1):.3f} writes={writes:.0f}"
        ),
    }
