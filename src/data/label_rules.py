"""Shared observable rules for heuristic labels and benchmark checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DIMENSION_NAMES = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
    "healthy",
]
BOTTLENECK_DIMENSIONS = DIMENSION_NAMES[:7]

SMALL_REQUEST_SHARE = 0.10
SMALL_REQUEST_COUNT = 1000
RANDOM_REQUEST_SHARE = 0.20
RANDOM_REQUEST_COUNT = 1000
COLLECTIVE_MIN_OPS = 1000
COLLECTIVE_SHARE = 0.50
METADATA_TIME_SHARE = 0.10
FSYNC_PER_WRITE = 0.50
RANK_IMBALANCE_RANGE_RATIO = 0.30
SHARED_STRAGGLER_SHARE = 0.15

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


def _mpiio_ops(frame: pd.DataFrame) -> pd.Series:
    return sum(
        frame[name]
        for name in (
            "MPIIO_INDEP_READS", "MPIIO_INDEP_WRITES",
            "MPIIO_COLL_READS", "MPIIO_COLL_WRITES",
            "MPIIO_NB_READS", "MPIIO_NB_WRITES",
        )
    )


def parallelism_present(features: pd.DataFrame | Mapping[str, object]):
    """Evaluate the rank-distribution rule without requiring other features."""
    frame = _as_frame(features)
    result = (
        (frame["nprocs"] > 1)
        & (
            (frame["rank_byte_range_ratio"] > RANK_IMBALANCE_RANGE_RATIO)
            | (frame["SHARED_BYTE_IMBALANCE"] > SHARED_STRAGGLER_SHARE)
        )
    )
    if isinstance(features, pd.DataFrame):
        return result
    return bool(result.iloc[0])


def rule_frame(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Evaluate the seven taxonomy rules on one or more feature rows."""
    frame = _as_frame(features)
    mpiio_ops = _mpiio_ops(frame)
    use_mpiio = mpiio_ops > 0

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
        operation_count = np.where(use_mpiio, mpiio_direction_ops, posix_ops)
        small_count = np.where(use_mpiio, mpiio_small, posix_small)
        small_direction.append(
            (small_count > SMALL_REQUEST_COUNT)
            & (small_count / np.maximum(operation_count, 1) > SMALL_REQUEST_SHARE)
        )

    random_direction = []
    for direction in ("READ", "WRITE"):
        operations = frame[f"POSIX_{direction}S"]
        random_operations = operations - frame[f"POSIX_SEQ_{direction}S"]
        random_direction.append(
            (random_operations > RANDOM_REQUEST_COUNT)
            & (random_operations / np.maximum(operations, 1) > RANDOM_REQUEST_SHARE)
        )

    collective = frame["MPIIO_COLL_READS"] + frame["MPIIO_COLL_WRITES"]
    mpiio_interface = (
        (mpiio_ops > COLLECTIVE_MIN_OPS)
        & (collective / np.maximum(mpiio_ops, 1) < COLLECTIVE_SHARE)
    )
    posix_interface = (
        (~use_mpiio)
        & (frame["nprocs"] > 1)
        & (frame["is_shared_file"] == 1)
    )
    writes = frame["POSIX_WRITES"]
    syncs = frame["POSIX_FSYNCS"] + frame["POSIX_FDSYNCS"]

    rules = pd.DataFrame(index=frame.index)
    rules["access_granularity"] = small_direction[0] | small_direction[1]
    rules["metadata_intensity"] = (
        (frame["metadata_time_ratio_all"] > METADATA_TIME_SHARE)
        | (frame["io_bytes_all"] == 0)
    )
    rules["parallelism_efficiency"] = parallelism_present(frame)
    rules["access_pattern"] = random_direction[0] | random_direction[1]
    rules["interface_choice"] = mpiio_interface | posix_interface
    rules["file_strategy"] = (
        (frame["nprocs"] > 1) & (frame["num_data_files"] >= frame["nprocs"])
    )
    rules["throughput_utilization"] = (
        (writes > 0) & (syncs / np.maximum(writes, 1) >= FSYNC_PER_WRITE)
    )
    return rules.astype(bool)


def labels_from_features(features: pd.DataFrame | Mapping[str, object]) -> pd.DataFrame:
    """Return the eight binary labels from the shared rule set."""
    rules = rule_frame(features)
    labels = rules.astype(int)
    labels["healthy"] = (~rules.any(axis=1)).astype(int)
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
    validity["interface_choice"] = ~(posix | mpiio)
    validity["file_strategy"] = ~(posix | stdio)
    validity["throughput_utilization"] = ~posix
    validity["healthy"] = validity[BOTTLENECK_DIMENSIONS].all(axis=1)
    return validity[DIMENSION_NAMES].astype(int)


def rule_details(features: Mapping[str, object]) -> dict[str, str]:
    """Human-readable values for one row, keyed by bottleneck dimension."""
    f = features
    mpiio_ops = sum(float(f[name]) for name in (
        "MPIIO_INDEP_READS", "MPIIO_INDEP_WRITES", "MPIIO_COLL_READS",
        "MPIIO_COLL_WRITES", "MPIIO_NB_READS", "MPIIO_NB_WRITES",
    ))
    layer = "mpiio" if mpiio_ops > 0 else "posix"
    small_parts = []
    for direction in ("READ", "WRITE"):
        if layer == "mpiio":
            operations = sum(float(f[f"MPIIO_{kind}_{direction}S"])
                             for kind in ("INDEP", "COLL", "NB"))
            small = sum(float(f[name]) for name in _MPIIO_SMALL[direction])
        else:
            operations = float(f[f"POSIX_{direction}S"])
            small = sum(float(f[name]) for name in _POSIX_SMALL[direction])
        small_parts.append(f"{direction.lower()}_small={small:.0f}/{operations:.0f}")
    posix_ops = float(f["POSIX_READS"]) + float(f["POSIX_WRITES"])
    random_ops = (
        posix_ops - float(f["POSIX_SEQ_READS"]) - float(f["POSIX_SEQ_WRITES"])
    )
    collective = float(f["MPIIO_COLL_READS"]) + float(f["MPIIO_COLL_WRITES"])
    writes = float(f["POSIX_WRITES"])
    syncs = float(f["POSIX_FSYNCS"]) + float(f["POSIX_FDSYNCS"])
    return {
        "access_granularity": f"{layer} {' '.join(small_parts)}",
        "metadata_intensity": (
            f"metadata_time_ratio_all={float(f['metadata_time_ratio_all']):.3f} "
            f"bytes={float(f['io_bytes_all']):.0f}"
        ),
        "parallelism_efficiency": (
            f"range_ratio={float(f['rank_byte_range_ratio']):.3f} "
            f"shared_imb={float(f['SHARED_BYTE_IMBALANCE']):.3f} "
            f"nprocs={float(f['nprocs']):.0f}"
        ),
        "access_pattern": f"random_posix={random_ops:.0f}/{posix_ops:.0f}",
        "interface_choice": (
            f"mpiio_ops={mpiio_ops:.0f} collective={collective:.0f}"
            if mpiio_ops > 0 else
            f"posix_shared={int(f['is_shared_file'])} nprocs={float(f['nprocs']):.0f}"
        ),
        "file_strategy": (
            f"data_files={float(f['num_data_files']):.0f} "
            f"nprocs={float(f['nprocs']):.0f}"
        ),
        "throughput_utilization": (
            f"syncs_per_write={syncs / max(writes, 1):.3f} writes={writes:.0f}"
        ),
    }
