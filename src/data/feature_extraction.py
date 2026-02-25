"""
Feature Extraction from Darshan Counters
=========================================
Converts raw Darshan counter dictionaries into a structured feature vector
for ML classification.

Produces ~136 features per job across these groups:
  - Job metadata (2): nprocs, runtime
  - POSIX raw counters (~85): operations, bytes, patterns, histograms, alignment, timing, imbalance
  - MPI-IO raw counters (~18): operations, bytes, timing
  - STDIO raw counters (~11): operations, bytes, timing
  - Derived ratios (~22): bandwidth, size ratios, pattern ratios, metadata, imbalance, temporal

Missing modules (e.g., MPI-IO absent for non-MPI jobs) are zero-filled.
Count/byte features are log10(x+1) transformed following AIIO (HPDC 2023).
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
# Feature definition lists
# ---------------------------------------------------------------------------

# All raw POSIX counters we extract (integer)
POSIX_INT_COUNTERS = [
    'POSIX_OPENS', 'POSIX_FILENOS', 'POSIX_DUPS',
    'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
    'POSIX_MMAPS', 'POSIX_FSYNCS', 'POSIX_FDSYNCS',
    'POSIX_RENAME_SOURCES', 'POSIX_RENAME_TARGETS',
    'POSIX_RW_SWITCHES',
    # Bytes
    'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
    'POSIX_MAX_BYTE_READ', 'POSIX_MAX_BYTE_WRITTEN',
    # Patterns
    'POSIX_CONSEC_READS', 'POSIX_CONSEC_WRITES',
    'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
    # Strides
    'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE',
    'POSIX_STRIDE3_STRIDE', 'POSIX_STRIDE4_STRIDE',
    'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
    'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT',
    # Size histogram (read)
    'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
    'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_10K_100K',
    'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_READ_1M_4M',
    'POSIX_SIZE_READ_4M_10M', 'POSIX_SIZE_READ_10M_100M',
    'POSIX_SIZE_READ_100M_1G', 'POSIX_SIZE_READ_1G_PLUS',
    # Size histogram (write)
    'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
    'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K',
    'POSIX_SIZE_WRITE_100K_1M', 'POSIX_SIZE_WRITE_1M_4M',
    'POSIX_SIZE_WRITE_4M_10M', 'POSIX_SIZE_WRITE_10M_100M',
    'POSIX_SIZE_WRITE_100M_1G', 'POSIX_SIZE_WRITE_1G_PLUS',
    # Access sizes
    'POSIX_ACCESS1_ACCESS', 'POSIX_ACCESS2_ACCESS',
    'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
    'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT',
    'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT',
    # Alignment
    'POSIX_MEM_NOT_ALIGNED', 'POSIX_MEM_ALIGNMENT',
    'POSIX_FILE_NOT_ALIGNED', 'POSIX_FILE_ALIGNMENT',
    # Rank imbalance
    'POSIX_FASTEST_RANK_BYTES', 'POSIX_SLOWEST_RANK_BYTES',
    'POSIX_MAX_READ_TIME_SIZE', 'POSIX_MAX_WRITE_TIME_SIZE',
]

# POSIX float counters
POSIX_FLOAT_COUNTERS = [
    'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
    'POSIX_F_MAX_READ_TIME', 'POSIX_F_MAX_WRITE_TIME',
    'POSIX_F_OPEN_START_TIMESTAMP', 'POSIX_F_CLOSE_END_TIMESTAMP',
    'POSIX_F_FASTEST_RANK_TIME', 'POSIX_F_SLOWEST_RANK_TIME',
    'POSIX_F_VARIANCE_RANK_TIME', 'POSIX_F_VARIANCE_RANK_BYTES',
]

# MPI-IO counters
MPIIO_INT_COUNTERS = [
    'MPIIO_INDEP_OPENS', 'MPIIO_COLL_OPENS',
    'MPIIO_INDEP_READS', 'MPIIO_INDEP_WRITES',
    'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES',
    'MPIIO_SPLIT_READS', 'MPIIO_SPLIT_WRITES',
    'MPIIO_NB_READS', 'MPIIO_NB_WRITES',
    'MPIIO_SYNCS', 'MPIIO_HINTS', 'MPIIO_VIEWS',
    'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
]

MPIIO_FLOAT_COUNTERS = [
    'MPIIO_F_READ_TIME', 'MPIIO_F_WRITE_TIME', 'MPIIO_F_META_TIME',
]

# STDIO counters
STDIO_INT_COUNTERS = [
    'STDIO_OPENS', 'STDIO_FDOPENS',
    'STDIO_READS', 'STDIO_WRITES', 'STDIO_SEEKS', 'STDIO_FLUSHES',
    'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
]

STDIO_FLOAT_COUNTERS = [
    'STDIO_F_READ_TIME', 'STDIO_F_WRITE_TIME', 'STDIO_F_META_TIME',
]

# All raw counter names (for reference)
ALL_RAW_COUNTERS = (
    POSIX_INT_COUNTERS + POSIX_FLOAT_COUNTERS
    + MPIIO_INT_COUNTERS + MPIIO_FLOAT_COUNTERS
    + STDIO_INT_COUNTERS + STDIO_FLOAT_COUNTERS
)

# Counters that should receive log10(x+1) transformation
# (count and byte features with heavy-tailed distributions)
LOG_TRANSFORM_COUNTERS = set(
    POSIX_INT_COUNTERS + MPIIO_INT_COUNTERS + STDIO_INT_COUNTERS
)

# Counters that should NOT be log-transformed (timing, variance, ratios)
NO_LOG_TRANSFORM = set(
    POSIX_FLOAT_COUNTERS + MPIIO_FLOAT_COUNTERS + STDIO_FLOAT_COUNTERS
)

# Derived feature names
DERIVED_FEATURE_NAMES = [
    # Bandwidth
    'read_bw_mb_s', 'write_bw_mb_s', 'total_bw_mb_s',
    # Size ratios
    'avg_read_size', 'avg_write_size',
    'small_read_ratio', 'small_write_ratio', 'small_io_ratio',
    'large_read_ratio', 'large_write_ratio',
    # Pattern ratios
    'seq_read_ratio', 'seq_write_ratio',
    'consec_read_ratio', 'consec_write_ratio',
    'rw_ratio', 'rw_switch_ratio',
    # Alignment ratios
    'mem_misalign_ratio', 'file_misalign_ratio',
    # Metadata ratios
    'metadata_time_ratio', 'opens_per_op', 'stats_per_op',
    'seeks_per_op', 'fsync_ratio', 'opens_per_mb',
    # Imbalance ratios
    'rank_bytes_cv', 'rank_time_cv',
    'byte_imbalance', 'time_imbalance',
    # MPI-IO ratios
    'collective_ratio', 'nonblocking_ratio',
    # Temporal
    'io_duration', 'io_fraction',
    # Interface
    'uses_mpiio', 'uses_stdio', 'num_files',
    # Access concentration
    'access_size_concentration', 'dominant_access_size',
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(parsed_log, apply_log_transform=True):
    """Extract full feature vector from a parsed Darshan log.

    Parameters
    ----------
    parsed_log : dict
        Output of ``parse_darshan_log()``.  Must contain keys
        ``'job'``, ``'counters'``, ``'modules'``.
    apply_log_transform : bool
        If True, apply log10(x+1) to count/byte features.

    Returns
    -------
    dict
        Feature dictionary with ~136 keys, ready for conversion to DataFrame row.
    """
    job = parsed_log['job']
    raw = parsed_log['counters']
    modules = parsed_log['modules']

    features = {}

    # --- Job metadata ---
    features['nprocs'] = max(job.get('nprocs', 1), 1)
    features['runtime_seconds'] = max(job.get('runtime', 0.0), 0.0)

    # --- Raw counters (zero-fill missing) ---
    for counter in ALL_RAW_COUNTERS:
        val = raw.get(counter, 0.0)
        # Replace Darshan sentinel -1 with 0
        if val == _SENTINEL:
            val = 0.0
        features[counter] = val

    # --- Derived features ---
    _compute_derived_features(features, modules)

    # --- Log transform on count/byte features ---
    if apply_log_transform:
        for key in LOG_TRANSFORM_COUNTERS:
            if key in features:
                features[key] = math.log10(max(features[key], 0) + 1)
        # Also transform job metadata
        features['nprocs'] = math.log10(features['nprocs'] + 1)
        features['runtime_seconds'] = math.log10(max(features['runtime_seconds'], 0) + 1)

    # --- Job info (not features, but carried for identification) ---
    features['_jobid'] = job.get('jobid', 0)
    features['_uid'] = job.get('uid', 0)
    features['_start_time'] = job.get('start_time', 0)
    features['_modules'] = ','.join(modules)
    features['_uses_lustre'] = 1 if job.get('uses_lustre', False) else 0

    return features


def get_feature_names(include_derived=True, include_metadata=True):
    """Return ordered list of feature column names (excluding _* info columns)."""
    names = []
    if include_metadata:
        names.extend(['nprocs', 'runtime_seconds'])
    names.extend(ALL_RAW_COUNTERS)
    if include_derived:
        names.extend(DERIVED_FEATURE_NAMES)
    return names


def get_info_columns():
    """Return list of _* info column names (not used as features)."""
    return ['_jobid', '_uid', '_start_time', '_modules', '_uses_lustre']


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def _compute_derived_features(f, modules):
    """Compute derived features in-place from raw counters.

    Parameters
    ----------
    f : dict
        Feature dictionary (modified in-place).  Contains raw counter values
        (NOT yet log-transformed).
    modules : list
        List of modules present in this log.
    """
    # Helper: safe get
    def g(key, default=0.0):
        val = f.get(key, default)
        return val if val != _SENTINEL else 0.0

    total_reads = g('POSIX_READS')
    total_writes = g('POSIX_WRITES')
    total_ops = total_reads + total_writes
    total_bytes = g('POSIX_BYTES_READ') + g('POSIX_BYTES_WRITTEN')
    read_time = g('POSIX_F_READ_TIME')
    write_time = g('POSIX_F_WRITE_TIME')
    meta_time = g('POSIX_F_META_TIME')
    total_time = read_time + write_time + meta_time

    nprocs = f.get('nprocs', 1)

    # --- Bandwidth ---
    f['read_bw_mb_s'] = g('POSIX_BYTES_READ') / max(read_time, _EPS) / 1e6
    f['write_bw_mb_s'] = g('POSIX_BYTES_WRITTEN') / max(write_time, _EPS) / 1e6
    io_time = read_time + write_time
    f['total_bw_mb_s'] = total_bytes / max(io_time, _EPS) / 1e6

    # --- Size ratios ---
    f['avg_read_size'] = g('POSIX_BYTES_READ') / max(total_reads, 1)
    f['avg_write_size'] = g('POSIX_BYTES_WRITTEN') / max(total_writes, 1)

    small_r = g('POSIX_SIZE_READ_0_100') + g('POSIX_SIZE_READ_100_1K')
    small_w = g('POSIX_SIZE_WRITE_0_100') + g('POSIX_SIZE_WRITE_100_1K')
    f['small_read_ratio'] = small_r / max(total_reads, 1)
    f['small_write_ratio'] = small_w / max(total_writes, 1)
    f['small_io_ratio'] = (small_r + small_w) / max(total_ops, 1)

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
    f['opens_per_op'] = g('POSIX_OPENS') / max(total_ops, 1)
    f['stats_per_op'] = g('POSIX_STATS') / max(total_ops, 1)
    f['seeks_per_op'] = g('POSIX_SEEKS') / max(total_ops, 1)
    f['fsync_ratio'] = g('POSIX_FSYNCS') / max(total_writes, 1)
    f['opens_per_mb'] = g('POSIX_OPENS') / max(total_bytes / 1e6, _EPS)

    # --- Imbalance ratios ---
    # Coefficient of variation (only meaningful for multi-rank jobs)
    var_bytes = max(g('POSIX_F_VARIANCE_RANK_BYTES'), 0)
    var_time = max(g('POSIX_F_VARIANCE_RANK_TIME'), 0)
    mean_bytes_per_rank = total_bytes / max(nprocs, 1)
    mean_time_per_rank = total_time / max(nprocs, 1)
    f['rank_bytes_cv'] = math.sqrt(var_bytes) / max(mean_bytes_per_rank, _EPS)
    f['rank_time_cv'] = math.sqrt(var_time) / max(mean_time_per_rank, _EPS)

    # Direct imbalance ratio
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
    runtime = f.get('runtime_seconds', 0)
    f['io_fraction'] = f['io_duration'] / max(runtime, _EPS)

    # --- Interface indicators ---
    f['uses_mpiio'] = 1 if 'MPI-IO' in modules else 0
    f['uses_stdio'] = 1 if 'STDIO' in modules else 0
    f['num_files'] = f.get('num_files', 0)

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
        config_path = Path(__file__).resolve().parents[2] / 'configs' / 'feature_extraction.yaml'
    with open(config_path) as fh:
        return yaml.safe_load(fh)
