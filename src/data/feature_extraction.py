"""
Feature Extraction from Darshan Counters
=========================================
Converts raw Darshan counter dictionaries into a structured feature vector
for ML classification.

``extract_raw_features`` returns all counters, metadata and indicators with no
transform (Stage 1, the immutable parquet). Derived features are computed once,
vectorized, in ``src.data.preprocessing.stage3_engineer`` (also for single logs
through ``engineer_one``); ``compute_layer_and_rank_features`` here holds the
part of that computation that this module defines.

Feature groups:
  - Job metadata: nprocs, runtime
  - POSIX raw counters: operations, bytes, patterns, histograms,
    alignment, timing, timestamps, imbalance
  - MPI-IO raw counters: operations, bytes, timing
  - STDIO raw counters: operations, bytes, timing
  - Module and file indicators: presence, partial state, sharing
  - Derived values: bandwidth, size, pattern, metadata, imbalance,
    temporal, access concentration

Missing modules (e.g., MPI-IO absent for non-MPI jobs) are zero-filled.
Feature exclusion and normalization are deferred to the preprocessing stage
and driven by statistical analysis, not hardcoded here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-9  # Avoid division by zero

# Sentinel value in Darshan for "not available"
_SENTINEL = -1

# Bumped whenever the set or meaning of raw columns changes. Stored in every
# raw row as ``_schema_version`` and checked by the preprocessing stages, so a
# parquet written by older code cannot be engineered into false zeros.
# 1: SC 2026 dataset (156 columns). 2: per-rank and shared-record statistics,
# variance counters kept for one shared file, MPI-IO request-size histograms,
# Lustre info columns removed. 3: incomplete module records were rejected.
# 4: incomplete-module indicators, exact data-file counts and stream-free
# rank totals were added; present modules require their complete counter schema.
FEATURE_SCHEMA_VERSION = 4

# ---------------------------------------------------------------------------
# Feature definition lists: ALL counters, no exclusions
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
    # Size histogram: read (SUM), 10 bins
    'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
    'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_10K_100K',
    'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_READ_1M_4M',
    'POSIX_SIZE_READ_4M_10M', 'POSIX_SIZE_READ_10M_100M',
    'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS',
    # Size histogram: write (SUM), 10 bins
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
    # Rank imbalance (CONDITIONAL: sentinel -1 if not shared)
    'POSIX_FASTEST_RANK', 'POSIX_FASTEST_RANK_BYTES',
    'POSIX_SLOWEST_RANK', 'POSIX_SLOWEST_RANK_BYTES',
    # Worst-case I/O size (CONDITIONAL)
    'POSIX_MAX_READ_TIME_SIZE', 'POSIX_MAX_WRITE_TIME_SIZE',
    # File metadata (LAST_VALUE: not meaningful as aggregate, kept for EDA)
    'POSIX_MODE', 'POSIX_RENAMED_FROM',
]

# POSIX float counters: includes ALL 8 timestamps
POSIX_FLOAT_COUNTERS = [
    # Cumulative I/O times (SUM)
    'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
    # Worst-case single-op times (MAX)
    'POSIX_F_MAX_READ_TIME', 'POSIX_F_MAX_WRITE_TIME',
    # Timestamps: START (MIN_NONZERO)
    'POSIX_F_OPEN_START_TIMESTAMP',
    'POSIX_F_READ_START_TIMESTAMP',
    'POSIX_F_WRITE_START_TIMESTAMP',
    'POSIX_F_CLOSE_START_TIMESTAMP',
    # Timestamps: END (MAX)
    'POSIX_F_OPEN_END_TIMESTAMP',
    'POSIX_F_READ_END_TIMESTAMP',
    'POSIX_F_WRITE_END_TIMESTAMP',
    'POSIX_F_CLOSE_END_TIMESTAMP',
    # Rank time imbalance (CONDITIONAL: 0.0 if not shared)
    'POSIX_F_FASTEST_RANK_TIME', 'POSIX_F_SLOWEST_RANK_TIME',
    # Variance of rank time and bytes for a shared record
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
    # Aggregate request-size histograms (SUM), the application's view of its
    # request sizes when it uses MPI-IO (POSIX sees ROMIO's chunks)
    'MPIIO_SIZE_READ_AGG_0_100', 'MPIIO_SIZE_READ_AGG_100_1K',
    'MPIIO_SIZE_READ_AGG_1K_10K', 'MPIIO_SIZE_READ_AGG_10K_100K',
    'MPIIO_SIZE_READ_AGG_100K_1M', 'MPIIO_SIZE_READ_AGG_1M_4M',
    'MPIIO_SIZE_READ_AGG_4M_10M', 'MPIIO_SIZE_READ_AGG_10M_100M',
    'MPIIO_SIZE_READ_AGG_100M_1G', 'MPIIO_SIZE_READ_AGG_1G_PLUS',
    'MPIIO_SIZE_WRITE_AGG_0_100', 'MPIIO_SIZE_WRITE_AGG_100_1K',
    'MPIIO_SIZE_WRITE_AGG_1K_10K', 'MPIIO_SIZE_WRITE_AGG_10K_100K',
    'MPIIO_SIZE_WRITE_AGG_100K_1M', 'MPIIO_SIZE_WRITE_AGG_1M_4M',
    'MPIIO_SIZE_WRITE_AGG_4M_10M', 'MPIIO_SIZE_WRITE_AGG_10M_100M',
    'MPIIO_SIZE_WRITE_AGG_100M_1G', 'MPIIO_SIZE_WRITE_AGG_1G_PLUS',
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

# Per-rank statistics computed from the file records before aggregation
# (parse_darshan._rank_statistics). Bytes and time are POSIX + STDIO.
RANK_STAT_COUNTERS = [
    'RANK_IO_COUNT',       # ranks with any I/O bytes or time
    'RANK_BYTES_MAX',      # bytes of the rank that moved the most
    'RANK_BYTES_MIN',      # bytes of the rank that moved the least (0 if idle)
    'RANK_BYTES_VAR',      # population variance of bytes over all ranks
    'RANK_BYTES_GINI',     # Gini coefficient of bytes over all ranks
    'RANK_TIME_MAX',       # I/O time of the slowest rank
    'RANK_TIME_MIN',       # I/O time of the fastest rank
    'RANK_TIME_VAR',       # population variance of I/O time over all ranks
    'RANK_BYTES_TOTAL',     # stream-free bytes represented by rank statistics
    'RANK_TIME_TOTAL',      # stream-free time represented by rank statistics
    'RANK_SHARED_BYTES',   # bytes in shared (rank -1) records
    'SHARED_BYTE_IMBALANCE',  # Drishti P18 over reduced records: |slowest - fastest bytes| / bytes,
    'SHARED_TIME_IMBALANCE',  # and P19 on time; from the MPI-IO records when the job
                              # has them (collective buffering makes POSIX uneven), else POSIX
    'FILE_WRITE_IMBALANCE',  # Drishti per-file (max - min) / max, written bytes
    'FILE_READ_IMBALANCE',   # same for read bytes
    'SHARED_POSIX_READS', 'SHARED_POSIX_WRITES',
    'SHARED_POSIX_SMALL_READS', 'SHARED_POSIX_SMALL_WRITES',
]

# All raw counter names
ALL_RAW_COUNTERS = (
    POSIX_INT_COUNTERS + POSIX_FLOAT_COUNTERS
    + MPIIO_INT_COUNTERS + MPIIO_FLOAT_COUNTERS
    + STDIO_INT_COUNTERS + STDIO_FLOAT_COUNTERS
    + RANK_STAT_COUNTERS
)

# ---------------------------------------------------------------------------
# Feature groups: for group-specific normalization in preprocessing
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
        'MPIIO_SIZE_READ_AGG_0_100', 'MPIIO_SIZE_READ_AGG_100_1K',
        'MPIIO_SIZE_READ_AGG_1K_10K', 'MPIIO_SIZE_READ_AGG_10K_100K',
        'MPIIO_SIZE_READ_AGG_100K_1M', 'MPIIO_SIZE_READ_AGG_1M_4M',
        'MPIIO_SIZE_READ_AGG_4M_10M', 'MPIIO_SIZE_READ_AGG_10M_100M',
        'MPIIO_SIZE_READ_AGG_100M_1G', 'MPIIO_SIZE_READ_AGG_1G_PLUS',
        'MPIIO_SIZE_WRITE_AGG_0_100', 'MPIIO_SIZE_WRITE_AGG_100_1K',
        'MPIIO_SIZE_WRITE_AGG_1K_10K', 'MPIIO_SIZE_WRITE_AGG_10K_100K',
        'MPIIO_SIZE_WRITE_AGG_100K_1M', 'MPIIO_SIZE_WRITE_AGG_1M_4M',
        'MPIIO_SIZE_WRITE_AGG_4M_10M', 'MPIIO_SIZE_WRITE_AGG_10M_100M',
        'MPIIO_SIZE_WRITE_AGG_100M_1G', 'MPIIO_SIZE_WRITE_AGG_1G_PLUS',
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
        'partial_posix', 'partial_mpiio', 'partial_stdio', 'is_shared_file',
    ],
    # Per-rank statistics (unbounded, heavy-tailed): log1p
    'rank_stat': [
        'RANK_IO_COUNT', 'RANK_BYTES_MAX', 'RANK_BYTES_MIN', 'RANK_BYTES_VAR',
        'RANK_TIME_MAX', 'RANK_TIME_MIN', 'RANK_TIME_VAR',
        'RANK_BYTES_TOTAL', 'RANK_TIME_TOTAL', 'RANK_SHARED_BYTES',
        'SHARED_POSIX_READS', 'SHARED_POSIX_WRITES',
        'SHARED_POSIX_SMALL_READS', 'SHARED_POSIX_SMALL_WRITES',
    ],
    # Per-rank statistics bounded in [0, 1]: no normalization
    'rank_stat_bounded': [
        'RANK_BYTES_GINI', 'SHARED_BYTE_IMBALANCE', 'SHARED_TIME_IMBALANCE',
        'FILE_WRITE_IMBALANCE', 'FILE_READ_IMBALANCE',
    ],
    # Derived ratios: bounded [0, 1], no normalization needed
    'ratio': [
        'read_ratio',
        'small_read_ratio', 'small_write_ratio', 'small_io_ratio',
        'medium_read_ratio', 'medium_write_ratio',
        'large_read_ratio', 'large_write_ratio',
        'seq_read_ratio', 'seq_write_ratio',
        'consec_read_ratio', 'consec_write_ratio',
        'rw_switch_ratio',
        'mem_misalign_ratio', 'file_misalign_ratio',
        'metadata_time_ratio', 'read_time_fraction', 'write_time_fraction',
        'byte_imbalance', 'time_imbalance',
        'collective_ratio', 'nonblocking_ratio',
        'access_size_concentration',
        # per-rank distribution (POSIX + STDIO records, all ranks)
        'io_rank_fraction', 'top_rank_byte_share', 'top_rank_time_share',
        'rank_byte_range_ratio', 'rank_time_range_ratio', 'shared_record_byte_share',
        # layer-agnostic (POSIX + STDIO) fractions
        'stdio_byte_share', 'metadata_time_ratio_all', 'write_time_fraction_all',
    ],
    # Derived unbounded: computed ratios/values with no fixed upper bound
    # These need log1p to compress their dynamic range
    'ratio_unbounded': [
        'read_bw_mb_s', 'write_bw_mb_s', 'total_bw_mb_s',
        'avg_read_size', 'avg_write_size',
        'rw_ratio',
        'opens_per_op', 'stats_per_op', 'seeks_per_op',
        'fsync_ratio', 'opens_per_mb',
        'rank_bytes_cv', 'rank_time_cv',
        'rank_bytes_cv_all', 'rank_time_cv_all',
        'avg_read_size_all', 'avg_write_size_all',
        'io_active_fraction',
    ],
    # Derived absolute: unbounded derived values
    'derived_absolute': [
        'io_duration', 'dominant_access_size', 'num_files', 'num_data_files',
        'io_bytes_all', 'io_ops_all',
    ],
    # Job metadata
    'metadata': [
        'nprocs', 'runtime_seconds',
    ],
}

# Info columns: carried for identification, never features
INFO_COLUMNS = [
    '_schema_version', '_jobid', '_uid', '_start_time', '_end_time',
    '_modules', '_log_version',
]

# Names produced by stage 3 on top of the raw columns (every derived group;
# file counts are emitted raw and therefore left out here)
DERIVED_FEATURE_NAMES = [
    name for name in (FEATURE_GROUPS['ratio'] + FEATURE_GROUPS['ratio_unbounded']
                      + FEATURE_GROUPS['derived_absolute'])
    if name not in {'num_files', 'num_data_files'}
]

MODULE_COUNTERS = {
    'POSIX': POSIX_INT_COUNTERS + POSIX_FLOAT_COUNTERS,
    'MPI-IO': MPIIO_INT_COUNTERS + MPIIO_FLOAT_COUNTERS,
    'STDIO': STDIO_INT_COUNTERS + STDIO_FLOAT_COUNTERS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_raw_features(parsed_log: dict[str, object]) -> dict[str, object]:
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
    if 'nprocs' not in job or int(job['nprocs']) < 1:
        raise ValueError(f"invalid or missing nprocs: {job.get('nprocs')}")
    if 'runtime' not in job:
        raise ValueError("parsed job lacks runtime")
    runtime = float(job['runtime'])
    if not np.isfinite(runtime) or runtime < 0:
        raise ValueError(f"invalid job runtime: {job['runtime']}")
    features['nprocs'] = int(job['nprocs'])
    features['runtime_seconds'] = runtime

    # --- Module counters: zero-fill absent modules, require present schemas ---
    for module, counters in MODULE_COUNTERS.items():
        if module in modules:
            missing = [counter for counter in counters if counter not in raw]
            if missing:
                raise ValueError(
                    f"present module {module} lacks {len(missing)} expected counters, "
                    f"first: {missing[:5]}"
                )
            for counter in counters:
                features[counter] = raw[counter]
        else:
            for counter in counters:
                features[counter] = 0.0
    if any(module in modules for module in MODULE_COUNTERS):
        missing = [counter for counter in RANK_STAT_COUNTERS if counter not in raw]
        if missing:
            raise ValueError(f"parsed rank statistics lack {missing[:5]}")
    for counter in RANK_STAT_COUNTERS:
        features[counter] = raw.get(counter, 0.0)

    # --- Module presence indicators ---
    features['has_posix'] = 1 if 'POSIX' in modules else 0
    features['has_mpiio'] = 1 if 'MPI-IO' in modules else 0
    features['has_stdio'] = 1 if 'STDIO' in modules else 0
    features['has_hdf5'] = 1 if ('H5F' in modules or 'H5D' in modules) else 0
    features['has_pnetcdf'] = 1 if 'PNETCDF' in modules else 0
    features['has_apmpi'] = 1 if 'APMPI' in modules else 0
    features['has_heatmap'] = 1 if 'HEATMAP' in modules else 0

    partial_modules = set(parsed_log.get('partial_modules', []))
    features['partial_posix'] = int('POSIX' in partial_modules)
    features['partial_mpiio'] = int('MPI-IO' in partial_modules)
    features['partial_stdio'] = int('STDIO' in partial_modules)

    # --- Shared file indicator (from parse_darshan shared_file_flag) ---
    # True if any POSIX file record is shared by several ranks.
    features['is_shared_file'] = 1 if shared_flags.get('POSIX', False) else 0

    # --- File counts ---
    if any(module in modules for module in MODULE_COUNTERS):
        for name in ('num_files', 'num_data_files'):
            if name not in raw:
                raise ValueError(f"parsed counters lack {name}")
    features['num_files'] = raw.get('num_files', 0)
    features['num_data_files'] = raw.get('num_data_files', 0)

    # --- Job info columns (not features, carried for identification) ---
    features['_schema_version'] = FEATURE_SCHEMA_VERSION
    features['_jobid'] = job.get('jobid', 0)
    features['_uid'] = job.get('uid', 0)
    features['_start_time'] = job.get('start_time', 0)
    features['_end_time'] = job.get('end_time', 0)
    features['_modules'] = ','.join(modules)
    features['_log_version'] = job.get('log_version', '')

    return features


def get_raw_feature_names() -> list[str]:
    """Ordered raw feature names: what ``extract_raw_features`` emits without
    the info columns."""
    return (['nprocs', 'runtime_seconds'] + ALL_RAW_COUNTERS
            + FEATURE_GROUPS['indicator'] + ['num_files', 'num_data_files'])


def get_feature_names() -> list[str]:
    """Ordered feature names after stage 3 (raw plus derived), without the
    info columns. ``tests/test_rank_statistics.py`` asserts that this equals
    the columns a real extraction produces."""
    return get_raw_feature_names() + DERIVED_FEATURE_NAMES


def get_info_columns() -> list[str]:
    """The ``_*`` identification columns (not features)."""
    return list(INFO_COLUMNS)


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def compute_layer_and_rank_features(g, nprocs):
    """Derived features over POSIX + STDIO and over the per-rank statistics.

    ``g(name)`` returns a counter as a float (one job) or as a column (many
    jobs); every expression below works on both, so the dict path and the
    vectorized path share this one implementation.

    Layer-agnostic totals exist because writers that use stdio (fwrite,
    fprintf, C++ streams) never reach the POSIX module, so the POSIX-only
    ratios are zero for them. Pattern ratios stay POSIX-only: STDIO has no
    size histogram or sequential counters.

    The per-rank features come from ``parse_darshan._rank_statistics``:
    ``top_rank_byte_share`` is 1.0 when one rank does all the I/O,
    ``io_rank_fraction`` is the share of ranks that did any I/O,
    ``rank_byte_range_ratio`` is Drishti's size-imbalance measure, (busiest
    rank minus idlest rank) / busiest rank, taken over all ranks of the job
    instead of the ranks of one file, and the ``*_cv_all`` values are the coefficients of variation over all
    ranks (ranks without I/O count as zero). The older ``byte_imbalance``,
    ``time_imbalance``, ``rank_bytes_cv`` and ``rank_time_cv`` describe a
    single shared file only.
    """
    bytes_read_all = g('POSIX_BYTES_READ') + g('STDIO_BYTES_READ')
    bytes_written_all = g('POSIX_BYTES_WRITTEN') + g('STDIO_BYTES_WRITTEN')
    bytes_all = bytes_read_all + bytes_written_all
    reads_all = g('POSIX_READS') + g('STDIO_READS')
    writes_all = g('POSIX_WRITES') + g('STDIO_WRITES')
    ops_all = reads_all + writes_all
    write_time_all = g('POSIX_F_WRITE_TIME') + g('STDIO_F_WRITE_TIME')
    meta_time_all = g('POSIX_F_META_TIME') + g('STDIO_F_META_TIME')
    time_all = g('POSIX_F_READ_TIME') + g('STDIO_F_READ_TIME') + write_time_all + meta_time_all
    rank_bytes_total = g('RANK_BYTES_TOTAL')
    rank_time_total = g('RANK_TIME_TOTAL')
    n = np.maximum(nprocs, 1)

    return {
        'io_bytes_all': bytes_all,
        'io_ops_all': ops_all,
        'avg_read_size_all': bytes_read_all / np.maximum(reads_all, 1),
        'avg_write_size_all': bytes_written_all / np.maximum(writes_all, 1),
        'stdio_byte_share': (g('STDIO_BYTES_READ') + g('STDIO_BYTES_WRITTEN'))
        / np.maximum(bytes_all, _EPS),
        'metadata_time_ratio_all': meta_time_all / np.maximum(time_all, _EPS),
        'write_time_fraction_all': write_time_all / np.maximum(time_all, _EPS),
        'io_rank_fraction': g('RANK_IO_COUNT') / n,
        'top_rank_byte_share': g('RANK_BYTES_MAX') / np.maximum(rank_bytes_total, _EPS),
        'top_rank_time_share': g('RANK_TIME_MAX') / np.maximum(rank_time_total, _EPS),
        'rank_byte_range_ratio': (g('RANK_BYTES_MAX') - g('RANK_BYTES_MIN'))
        / np.maximum(g('RANK_BYTES_MAX'), _EPS),
        'rank_time_range_ratio': (g('RANK_TIME_MAX') - g('RANK_TIME_MIN'))
        / np.maximum(g('RANK_TIME_MAX'), _EPS),
        'shared_record_byte_share': g('RANK_SHARED_BYTES') / np.maximum(rank_bytes_total, _EPS),
        'rank_bytes_cv_all': np.sqrt(np.maximum(g('RANK_BYTES_VAR'), 0))
        / np.maximum(rank_bytes_total / n, _EPS),
        'rank_time_cv_all': np.sqrt(np.maximum(g('RANK_TIME_VAR'), 0))
        / np.maximum(rank_time_total / n, _EPS),
    }
