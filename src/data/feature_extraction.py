"""
Feature Extraction from Darshan Counters
=========================================
Converts raw Darshan counter dictionaries into a structured feature vector
for ML classification.

Two modes of operation:
  1. **Raw extraction** (``extract_raw_features``): All counters + metadata +
     indicators, NO transforms.  Used for Stage 1 (immutable parquet).
  2. **Full extraction** (``extract_features``): Raw + derived ratios +
     optional log10(x+1) transform.  Backward-compatible API.

Feature groups (~150 total):
  - Job metadata (2): nprocs, runtime
  - POSIX raw counters (~85): operations, bytes, patterns, histograms,
    alignment, timing, timestamps, imbalance
  - MPI-IO raw counters (~18): operations, bytes, timing
  - STDIO raw counters (~11): operations, bytes, timing
  - Module/file indicators (~7): has_mpiio, has_stdio, has_hdf5, etc.
  - Derived ratios (~30): bandwidth, size, pattern, metadata, imbalance,
    temporal, access concentration

Missing modules (e.g., MPI-IO absent for non-MPI jobs) are zero-filled.
Feature exclusion and normalization are deferred to the preprocessing stage
and driven by statistical analysis, not hardcoded here.
"""

import logging
import math
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-9  # Avoid division by zero

# Sentinel value in Darshan for "not available"
_SENTINEL = -1

# ---------------------------------------------------------------------------
# Feature definition lists — ALL counters, no exclusions
# ---------------------------------------------------------------------------

# POSIX integer counters (SUM, MAX, LAST_VALUE, CONDITIONAL, TOP-4 MERGE)
POSIX_INT_COUNTERS = [
    # Operations (SUM)
    'POSIX_OPENS', 'POSIX_FILENOS', 'POSIX_DUPS',
    'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
    'POSIX_MMAPS', 'POSIX_FSYNCS', 'POSIX_FDSYNCS',
    'POSIX_RENAME_SOURCES', 'POSIX_RENAME_TARGETS',
    'POSIX_RW_SWITCHES',
    # Bytes (SUM / MAX)
    'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
    'POSIX_MAX_BYTE_READ', 'POSIX_MAX_BYTE_WRITTEN',
    # Patterns (SUM)
    'POSIX_CONSEC_READS', 'POSIX_CONSEC_WRITES',
    'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
    # Alignment (SUM for NOT_ALIGNED; LAST_VALUE for ALIGNMENT constants)
    'POSIX_MEM_NOT_ALIGNED', 'POSIX_MEM_ALIGNMENT',
    'POSIX_FILE_NOT_ALIGNED', 'POSIX_FILE_ALIGNMENT',
    # Size histogram — read (SUM), 10 bins
    'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
    'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_10K_100K',
    'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_READ_1M_4M',
    'POSIX_SIZE_READ_4M_10M', 'POSIX_SIZE_READ_10M_100M',
    'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS',
    # Size histogram — write (SUM), 10 bins
    'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
    'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K',
    'POSIX_SIZE_WRITE_100K_1M', 'POSIX_SIZE_WRITE_1M_4M',
    'POSIX_SIZE_WRITE_4M_10M', 'POSIX_SIZE_WRITE_10M_100M',
    'POSIX_SIZE_WRITE_100M_1G', 'POSIX_SIZE_WRITE_1G_PLUS',
    # Top-4 access sizes (TOP-4 MERGE)
    'POSIX_ACCESS1_ACCESS', 'POSIX_ACCESS2_ACCESS',
    'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
    'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT',
    'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT',
    # Top-4 strides (TOP-4 MERGE)
    'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE',
    'POSIX_STRIDE3_STRIDE', 'POSIX_STRIDE4_STRIDE',
    'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
    'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT',
    # Rank imbalance (CONDITIONAL — sentinel -1 if not shared)
    'POSIX_FASTEST_RANK', 'POSIX_FASTEST_RANK_BYTES',
    'POSIX_SLOWEST_RANK', 'POSIX_SLOWEST_RANK_BYTES',
    # Worst-case I/O size (CONDITIONAL)
    'POSIX_MAX_READ_TIME_SIZE', 'POSIX_MAX_WRITE_TIME_SIZE',
    # File metadata (LAST_VALUE — not meaningful as aggregate, kept for EDA)
    'POSIX_MODE', 'POSIX_RENAMED_FROM',
]

# POSIX float counters — includes ALL 8 timestamps
POSIX_FLOAT_COUNTERS = [
    # Cumulative I/O times (SUM)
    'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
    # Worst-case single-op times (MAX)
    'POSIX_F_MAX_READ_TIME', 'POSIX_F_MAX_WRITE_TIME',
    # Timestamps — START (MIN_NONZERO)
    'POSIX_F_OPEN_START_TIMESTAMP',
    'POSIX_F_READ_START_TIMESTAMP',
    'POSIX_F_WRITE_START_TIMESTAMP',
    'POSIX_F_CLOSE_START_TIMESTAMP',
    # Timestamps — END (MAX)
    'POSIX_F_OPEN_END_TIMESTAMP',
    'POSIX_F_READ_END_TIMESTAMP',
    'POSIX_F_WRITE_END_TIMESTAMP',
    'POSIX_F_CLOSE_END_TIMESTAMP',
    # Rank time imbalance (CONDITIONAL — 0.0 if not shared)
    'POSIX_F_FASTEST_RANK_TIME', 'POSIX_F_SLOWEST_RANK_TIME',
    # Variance (ZEROED in v3.5.0 --total)
    'POSIX_F_VARIANCE_RANK_TIME', 'POSIX_F_VARIANCE_RANK_BYTES',
]

# MPI-IO integer counters
MPIIO_INT_COUNTERS = [
    'MPIIO_INDEP_OPENS', 'MPIIO_COLL_OPENS',
    'MPIIO_INDEP_READS', 'MPIIO_INDEP_WRITES',
    'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES',
    'MPIIO_SPLIT_READS', 'MPIIO_SPLIT_WRITES',
    'MPIIO_NB_READS', 'MPIIO_NB_WRITES',
    'MPIIO_SYNCS', 'MPIIO_HINTS', 'MPIIO_VIEWS',
    'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
    'MPIIO_RW_SWITCHES',
    # Top-4 access sizes (TOP-4 MERGE)
    'MPIIO_ACCESS1_ACCESS', 'MPIIO_ACCESS2_ACCESS',
    'MPIIO_ACCESS3_ACCESS', 'MPIIO_ACCESS4_ACCESS',
    'MPIIO_ACCESS1_COUNT', 'MPIIO_ACCESS2_COUNT',
    'MPIIO_ACCESS3_COUNT', 'MPIIO_ACCESS4_COUNT',
]

# MPI-IO float counters
MPIIO_FLOAT_COUNTERS = [
    'MPIIO_F_READ_TIME', 'MPIIO_F_WRITE_TIME', 'MPIIO_F_META_TIME',
    'MPIIO_F_MAX_READ_TIME', 'MPIIO_F_MAX_WRITE_TIME',
    'MPIIO_F_FASTEST_RANK_TIME', 'MPIIO_F_SLOWEST_RANK_TIME',
    'MPIIO_F_VARIANCE_RANK_TIME', 'MPIIO_F_VARIANCE_RANK_BYTES',
]

# STDIO integer counters
STDIO_INT_COUNTERS = [
    'STDIO_OPENS', 'STDIO_FDOPENS',
    'STDIO_READS', 'STDIO_WRITES', 'STDIO_SEEKS', 'STDIO_FLUSHES',
    'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
    'STDIO_MAX_BYTE_READ', 'STDIO_MAX_BYTE_WRITTEN',
]

# STDIO float counters
STDIO_FLOAT_COUNTERS = [
    'STDIO_F_READ_TIME', 'STDIO_F_WRITE_TIME', 'STDIO_F_META_TIME',
    'STDIO_F_FASTEST_RANK_TIME', 'STDIO_F_SLOWEST_RANK_TIME',
    'STDIO_F_VARIANCE_RANK_TIME', 'STDIO_F_VARIANCE_RANK_BYTES',
]

# All raw counter names
ALL_RAW_COUNTERS = (
    POSIX_INT_COUNTERS + POSIX_FLOAT_COUNTERS
    + MPIIO_INT_COUNTERS + MPIIO_FLOAT_COUNTERS
    + STDIO_INT_COUNTERS + STDIO_FLOAT_COUNTERS
)

# ---------------------------------------------------------------------------
# Feature groups — for group-specific normalization in preprocessing
# ---------------------------------------------------------------------------
# These groups are used by preprocessing.py to apply different normalization
# strategies per counter type.  Feature exclusion is NOT done here; it is
# deferred to after EDA (statistical analysis of Stage 1 raw features).

FEATURE_GROUPS = {
    # Volume counters: extremely heavy-tailed, 0 to 10^15
    'volume': [
        'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
        'POSIX_MAX_BYTE_READ', 'POSIX_MAX_BYTE_WRITTEN',
        'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
        'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
        'STDIO_MAX_BYTE_READ', 'STDIO_MAX_BYTE_WRITTEN',
        'POSIX_FASTEST_RANK_BYTES', 'POSIX_SLOWEST_RANK_BYTES',
    ],
    # Operation counts: heavy-tailed, 0 to 10^9
    'count': [
        'POSIX_OPENS', 'POSIX_FILENOS', 'POSIX_DUPS',
        'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
        'POSIX_MMAPS', 'POSIX_FSYNCS', 'POSIX_FDSYNCS',
        'POSIX_RENAME_SOURCES', 'POSIX_RENAME_TARGETS',
        'POSIX_RW_SWITCHES',
        'POSIX_CONSEC_READS', 'POSIX_CONSEC_WRITES',
        'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
        'MPIIO_INDEP_OPENS', 'MPIIO_COLL_OPENS',
        'MPIIO_INDEP_READS', 'MPIIO_INDEP_WRITES',
        'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES',
        'MPIIO_SPLIT_READS', 'MPIIO_SPLIT_WRITES',
        'MPIIO_NB_READS', 'MPIIO_NB_WRITES',
        'MPIIO_SYNCS', 'MPIIO_HINTS', 'MPIIO_VIEWS',
        'MPIIO_RW_SWITCHES',
        'STDIO_OPENS', 'STDIO_FDOPENS',
        'STDIO_READS', 'STDIO_WRITES', 'STDIO_SEEKS', 'STDIO_FLUSHES',
    ],
    # Size histograms: sparse, non-negative integer counts
    'histogram': [
        'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
        'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_10K_100K',
        'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_READ_1M_4M',
        'POSIX_SIZE_READ_4M_10M', 'POSIX_SIZE_READ_10M_100M',
        'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS',
        'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
        'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K',
        'POSIX_SIZE_WRITE_100K_1M', 'POSIX_SIZE_WRITE_1M_4M',
        'POSIX_SIZE_WRITE_4M_10M', 'POSIX_SIZE_WRITE_10M_100M',
        'POSIX_SIZE_WRITE_100M_1G', 'POSIX_SIZE_WRITE_1G_PLUS',
    ],
    # Top-4 access/stride values and counts
    'top4': [
        'POSIX_ACCESS1_ACCESS', 'POSIX_ACCESS2_ACCESS',
        'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
        'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT',
        'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT',
        'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE',
        'POSIX_STRIDE3_STRIDE', 'POSIX_STRIDE4_STRIDE',
        'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
        'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT',
        'MPIIO_ACCESS1_ACCESS', 'MPIIO_ACCESS2_ACCESS',
        'MPIIO_ACCESS3_ACCESS', 'MPIIO_ACCESS4_ACCESS',
        'MPIIO_ACCESS1_COUNT', 'MPIIO_ACCESS2_COUNT',
        'MPIIO_ACCESS3_COUNT', 'MPIIO_ACCESS4_COUNT',
    ],
    # Timing: cumulative I/O times (seconds, SUM across records)
    'timing': [
        'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
        'POSIX_F_MAX_READ_TIME', 'POSIX_F_MAX_WRITE_TIME',
        'POSIX_F_FASTEST_RANK_TIME', 'POSIX_F_SLOWEST_RANK_TIME',
        'POSIX_F_VARIANCE_RANK_TIME', 'POSIX_F_VARIANCE_RANK_BYTES',
        'MPIIO_F_READ_TIME', 'MPIIO_F_WRITE_TIME', 'MPIIO_F_META_TIME',
        'MPIIO_F_MAX_READ_TIME', 'MPIIO_F_MAX_WRITE_TIME',
        'MPIIO_F_FASTEST_RANK_TIME', 'MPIIO_F_SLOWEST_RANK_TIME',
        'MPIIO_F_VARIANCE_RANK_TIME', 'MPIIO_F_VARIANCE_RANK_BYTES',
        'STDIO_F_READ_TIME', 'STDIO_F_WRITE_TIME', 'STDIO_F_META_TIME',
        'STDIO_F_FASTEST_RANK_TIME', 'STDIO_F_SLOWEST_RANK_TIME',
        'STDIO_F_VARIANCE_RANK_TIME', 'STDIO_F_VARIANCE_RANK_BYTES',
    ],
    # Timestamps: absolute time values (derive features, don't use directly)
    'timestamp': [
        'POSIX_F_OPEN_START_TIMESTAMP', 'POSIX_F_READ_START_TIMESTAMP',
        'POSIX_F_WRITE_START_TIMESTAMP', 'POSIX_F_CLOSE_START_TIMESTAMP',
        'POSIX_F_OPEN_END_TIMESTAMP', 'POSIX_F_READ_END_TIMESTAMP',
        'POSIX_F_WRITE_END_TIMESTAMP', 'POSIX_F_CLOSE_END_TIMESTAMP',
    ],
    # Categorical: system constants (LAST_VALUE aggregation)
    'categorical': [
        'POSIX_MODE', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
        'POSIX_RENAMED_FROM',
    ],
    # Rank IDs: integer rank indices (CONDITIONAL, sentinel -1)
    'rank_id': [
        'POSIX_FASTEST_RANK', 'POSIX_SLOWEST_RANK',
    ],
    # Conditional sizes: paired with F_MAX_*_TIME (CONDITIONAL)
    'conditional_size': [
        'POSIX_MAX_READ_TIME_SIZE', 'POSIX_MAX_WRITE_TIME_SIZE',
    ],
    # Binary indicators: 0 or 1
    'indicator': [
        'has_posix', 'has_mpiio', 'has_stdio',
        'has_hdf5', 'has_pnetcdf', 'has_apmpi', 'has_heatmap',
        'is_shared_file',
    ],
    # Derived ratios: bounded [0, 1] or [0, inf)
    'ratio': [
        'read_ratio', 'read_bw_mb_s', 'write_bw_mb_s', 'total_bw_mb_s',
        'avg_read_size', 'avg_write_size',
        'small_read_ratio', 'small_write_ratio', 'small_io_ratio',
        'medium_read_ratio', 'medium_write_ratio',
        'large_read_ratio', 'large_write_ratio',
        'seq_read_ratio', 'seq_write_ratio',
        'consec_read_ratio', 'consec_write_ratio',
        'rw_ratio', 'rw_switch_ratio',
        'mem_misalign_ratio', 'file_misalign_ratio',
        'metadata_time_ratio', 'read_time_fraction', 'write_time_fraction',
        'opens_per_op', 'stats_per_op', 'seeks_per_op',
        'fsync_ratio', 'opens_per_mb',
        'rank_bytes_cv', 'rank_time_cv',
        'byte_imbalance', 'time_imbalance',
        'collective_ratio', 'nonblocking_ratio',
        'io_active_fraction',
        'access_size_concentration',
    ],
    # Derived absolute: unbounded derived values
    'derived_absolute': [
        'io_duration', 'dominant_access_size', 'num_files',
    ],
    # Job metadata
    'metadata': [
        'nprocs', 'runtime_seconds',
    ],
}

# Counters that receive log10(x+1) in legacy mode (backward compat)
LOG_TRANSFORM_COUNTERS = set(
    FEATURE_GROUPS['volume'] + FEATURE_GROUPS['count']
    + FEATURE_GROUPS['histogram'] + FEATURE_GROUPS['top4']
)

# Derived feature names (all)
DERIVED_FEATURE_NAMES = (
    FEATURE_GROUPS['indicator']
    + FEATURE_GROUPS['ratio']
    + FEATURE_GROUPS['derived_absolute']
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_raw_features(parsed_log):
    """Extract ALL raw counters + indicators from a parsed Darshan log.

    This is Stage 1 extraction: no transforms, no exclusions, no derived
    ratios.  Saves the raw parsed values for later EDA and preprocessing.

    Parameters
    ----------
    parsed_log : dict
        Output of ``parse_darshan_log()``.  Must contain keys
        ``'job'``, ``'counters'``, ``'modules'``.

    Returns
    -------
    dict
        Feature dictionary with raw values + indicators + info columns.
    """
    job = parsed_log['job']
    raw = parsed_log['counters']
    modules = parsed_log['modules']
    shared_flags = parsed_log.get('shared_file_flags', {})

    features = {}

    # --- Job metadata ---
    features['nprocs'] = max(job.get('nprocs', 1), 1)
    features['runtime_seconds'] = max(job.get('runtime', 0.0), 0.0)

    # --- ALL raw counters (zero-fill if missing module) ---
    for counter in ALL_RAW_COUNTERS:
        features[counter] = raw.get(counter, 0.0)

    # --- Module presence indicators ---
    features['has_posix'] = 1 if 'POSIX' in modules else 0
    features['has_mpiio'] = 1 if 'MPI-IO' in modules else 0
    features['has_stdio'] = 1 if 'STDIO' in modules else 0
    features['has_hdf5'] = 1 if ('H5F' in modules or 'H5D' in modules) else 0
    features['has_pnetcdf'] = 1 if 'PNETCDF' in modules else 0
    features['has_apmpi'] = 1 if 'APMPI' in modules else 0
    features['has_heatmap'] = 1 if 'HEATMAP' in modules else 0

    # --- Shared file indicator (from parse_darshan shared_file_flag) ---
    # True if ALL POSIX records reference the same file ID
    features['is_shared_file'] = 1 if shared_flags.get('POSIX', False) else 0

    # --- File count ---
    features['num_files'] = raw.get('num_files', 0)

    # --- Job info columns (not features, carried for identification) ---
    features['_jobid'] = job.get('jobid', 0)
    features['_uid'] = job.get('uid', 0)
    features['_start_time'] = job.get('start_time', 0)
    features['_end_time'] = job.get('end_time', 0)
    features['_modules'] = ','.join(modules)
    features['_uses_lustre'] = 1 if job.get('uses_lustre', False) else 0
    features['_log_version'] = job.get('log_version', '')
    features['_lustre_mount'] = job.get('lustre_mount', '')

    return features


def extract_features(parsed_log, apply_log_transform=True):
    """Extract full feature vector from a parsed Darshan log.

    This combines raw extraction + derived feature computation + optional
    log10(x+1) transform.  Backward-compatible with existing pipeline.

    Parameters
    ----------
    parsed_log : dict
        Output of ``parse_darshan_log()``.
    apply_log_transform : bool
        If True, apply log10(x+1) to count/byte features.

    Returns
    -------
    dict
        Feature dictionary with ~150 keys.
    """
    features = extract_raw_features(parsed_log)
    modules = parsed_log['modules']

    # Replace sentinels with 0 for derived feature computation
    for key in list(features.keys()):
        if not key.startswith('_') and features[key] == _SENTINEL:
            features[key] = 0.0

    # Compute derived features
    _compute_derived_features(features, modules)

    # Apply log10(x+1) transform on heavy-tailed counters
    if apply_log_transform:
        for key in LOG_TRANSFORM_COUNTERS:
            if key in features:
                features[key] = math.log10(max(features[key], 0) + 1)
        # Also transform job metadata
        features['nprocs'] = math.log10(features['nprocs'] + 1)
        features['runtime_seconds'] = math.log10(
            max(features['runtime_seconds'], 0) + 1
        )

    return features


def get_feature_names(include_derived=True, include_metadata=True):
    """Return ordered list of feature column names (excluding _* info)."""
    names = []
    if include_metadata:
        names.extend(['nprocs', 'runtime_seconds'])
    names.extend(ALL_RAW_COUNTERS)
    if include_derived:
        names.extend(DERIVED_FEATURE_NAMES)
    return names


def get_info_columns():
    """Return list of _* info column names (not used as features)."""
    return [
        '_jobid', '_uid', '_start_time', '_end_time',
        '_modules', '_uses_lustre', '_log_version', '_lustre_mount',
    ]


def get_raw_feature_names():
    """Return ordered list of raw feature names (counters + indicators)."""
    names = ['nprocs', 'runtime_seconds']
    names.extend(ALL_RAW_COUNTERS)
    names.extend(FEATURE_GROUPS['indicator'])
    names.append('num_files')
    return names


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def _compute_derived_features(f, modules):
    """Compute derived features in-place from raw counters.

    Parameters
    ----------
    f : dict
        Feature dictionary (modified in-place).  Contains raw counter values
        with sentinels already replaced by 0.
    modules : list
        List of modules present in this log.
    """
    # Helper: safe get (0 if missing or sentinel)
    def g(key, default=0.0):
        val = f.get(key, default)
        return val if val != _SENTINEL else 0.0

    total_reads = g('POSIX_READS')
    total_writes = g('POSIX_WRITES')
    total_ops = total_reads + total_writes
    bytes_read = g('POSIX_BYTES_READ')
    bytes_written = g('POSIX_BYTES_WRITTEN')
    total_bytes = bytes_read + bytes_written
    read_time = g('POSIX_F_READ_TIME')
    write_time = g('POSIX_F_WRITE_TIME')
    meta_time = g('POSIX_F_META_TIME')
    total_time = read_time + write_time + meta_time
    nprocs = f.get('nprocs', 1)
    runtime = f.get('runtime_seconds', 0)

    # --- Read/write balance ---
    f['read_ratio'] = bytes_read / max(total_bytes, 1)

    # --- Bandwidth ---
    f['read_bw_mb_s'] = bytes_read / max(read_time, _EPS) / 1e6
    f['write_bw_mb_s'] = bytes_written / max(write_time, _EPS) / 1e6
    io_time = read_time + write_time
    f['total_bw_mb_s'] = total_bytes / max(io_time, _EPS) / 1e6

    # --- Average sizes ---
    f['avg_read_size'] = bytes_read / max(total_reads, 1)
    f['avg_write_size'] = bytes_written / max(total_writes, 1)

    # --- Size distribution ratios ---
    # Small I/O: < 1 KB
    small_r = g('POSIX_SIZE_READ_0_100') + g('POSIX_SIZE_READ_100_1K')
    small_w = g('POSIX_SIZE_WRITE_0_100') + g('POSIX_SIZE_WRITE_100_1K')
    f['small_read_ratio'] = small_r / max(total_reads, 1)
    f['small_write_ratio'] = small_w / max(total_writes, 1)
    f['small_io_ratio'] = (small_r + small_w) / max(total_ops, 1)

    # Medium I/O: 1 KB to 1 MB
    medium_r = (g('POSIX_SIZE_READ_1K_10K') + g('POSIX_SIZE_READ_10K_100K')
                + g('POSIX_SIZE_READ_100K_1M'))
    medium_w = (g('POSIX_SIZE_WRITE_1K_10K') + g('POSIX_SIZE_WRITE_10K_100K')
                + g('POSIX_SIZE_WRITE_100K_1M'))
    f['medium_read_ratio'] = medium_r / max(total_reads, 1)
    f['medium_write_ratio'] = medium_w / max(total_writes, 1)

    # Large I/O: >= 1 MB
    large_r = (g('POSIX_SIZE_READ_1M_4M') + g('POSIX_SIZE_READ_4M_10M')
               + g('POSIX_SIZE_READ_10M_100M') + g('POSIX_SIZE_READ_100M_1G')
               + g('POSIX_SIZE_READ_1G_PLUS'))
    large_w = (g('POSIX_SIZE_WRITE_1M_4M') + g('POSIX_SIZE_WRITE_4M_10M')
               + g('POSIX_SIZE_WRITE_10M_100M') + g('POSIX_SIZE_WRITE_100M_1G')
               + g('POSIX_SIZE_WRITE_1G_PLUS'))
    f['large_read_ratio'] = large_r / max(total_reads, 1)
    f['large_write_ratio'] = large_w / max(total_writes, 1)

    # --- Pattern ratios ---
    f['seq_read_ratio'] = g('POSIX_SEQ_READS') / max(total_reads, 1)
    f['seq_write_ratio'] = g('POSIX_SEQ_WRITES') / max(total_writes, 1)
    f['consec_read_ratio'] = g('POSIX_CONSEC_READS') / max(total_reads, 1)
    f['consec_write_ratio'] = g('POSIX_CONSEC_WRITES') / max(total_writes, 1)
    f['rw_ratio'] = total_reads / max(total_writes, 1)
    f['rw_switch_ratio'] = g('POSIX_RW_SWITCHES') / max(total_ops, 1)

    # --- Alignment ratios ---
    f['mem_misalign_ratio'] = g('POSIX_MEM_NOT_ALIGNED') / max(total_ops, 1)
    f['file_misalign_ratio'] = g('POSIX_FILE_NOT_ALIGNED') / max(total_ops, 1)

    # --- Metadata ratios ---
    f['metadata_time_ratio'] = meta_time / max(total_time, _EPS)
    f['read_time_fraction'] = read_time / max(total_time, _EPS)
    f['write_time_fraction'] = write_time / max(total_time, _EPS)
    f['opens_per_op'] = g('POSIX_OPENS') / max(total_ops, 1)
    f['stats_per_op'] = g('POSIX_STATS') / max(total_ops, 1)
    f['seeks_per_op'] = g('POSIX_SEEKS') / max(total_ops, 1)
    f['fsync_ratio'] = g('POSIX_FSYNCS') / max(total_writes, 1)
    f['opens_per_mb'] = g('POSIX_OPENS') / max(total_bytes / 1e6, _EPS)

    # --- Imbalance ratios ---
    var_bytes = max(g('POSIX_F_VARIANCE_RANK_BYTES'), 0)
    var_time = max(g('POSIX_F_VARIANCE_RANK_TIME'), 0)
    mean_bytes_per_rank = total_bytes / max(nprocs, 1)
    mean_time_per_rank = total_time / max(nprocs, 1)
    f['rank_bytes_cv'] = math.sqrt(var_bytes) / max(mean_bytes_per_rank, _EPS)
    f['rank_time_cv'] = math.sqrt(var_time) / max(mean_time_per_rank, _EPS)

    fastest_bytes = g('POSIX_FASTEST_RANK_BYTES')
    slowest_bytes = g('POSIX_SLOWEST_RANK_BYTES')
    f['byte_imbalance'] = (slowest_bytes - fastest_bytes) / max(total_bytes, _EPS)

    fastest_time = g('POSIX_F_FASTEST_RANK_TIME')
    slowest_time = g('POSIX_F_SLOWEST_RANK_TIME')
    f['time_imbalance'] = (slowest_time - fastest_time) / max(total_time, _EPS)

    # --- MPI-IO ratios ---
    coll = g('MPIIO_COLL_READS') + g('MPIIO_COLL_WRITES')
    indep = g('MPIIO_INDEP_READS') + g('MPIIO_INDEP_WRITES')
    nb = g('MPIIO_NB_READS') + g('MPIIO_NB_WRITES')
    total_mpiio = coll + indep + nb
    f['collective_ratio'] = coll / max(total_mpiio, 1)
    f['nonblocking_ratio'] = nb / max(total_mpiio, 1)

    # --- Temporal ---
    open_start = g('POSIX_F_OPEN_START_TIMESTAMP')
    close_end = g('POSIX_F_CLOSE_END_TIMESTAMP')
    f['io_duration'] = max(close_end - open_start, 0)
    f['io_active_fraction'] = total_time / max(runtime, _EPS)

    # --- Access concentration ---
    f['access_size_concentration'] = g('POSIX_ACCESS1_COUNT') / max(total_ops, 1)
    f['dominant_access_size'] = g('POSIX_ACCESS1_ACCESS')


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_feature_config(config_path=None):
    """Load feature extraction configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to feature_extraction.yaml.  Defaults to
        ``configs/feature_extraction.yaml`` relative to project root.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / 'configs' / 'feature_extraction.yaml'
        )
    with open(config_path) as fh:
        return yaml.safe_load(fh)
