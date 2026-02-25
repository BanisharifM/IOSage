"""
Job-Level Aggregation
=====================
Aggregates per-file Darshan records into a single feature vector per job.

Darshan logs contain one record per file per rank. For ML, we need one row
per job. This module handles the aggregation with appropriate strategies:

  - **Sum**: bytes, operation counts, time counters
  - **Max**: variance, slowest rank bytes/time, max byte offset
  - **Min**: fastest rank bytes/time, start timestamps
  - **Count**: unique files, unique modules

For shared files (rank = -1), the counters already represent the aggregate
across all ranks. For unique files (one per rank), we sum across ranks.
"""

import logging

logger = logging.getLogger(__name__)

# Counters that should be aggregated with MAX (not sum)
MAX_COUNTERS = {
    'POSIX_MAX_BYTE_READ', 'POSIX_MAX_BYTE_WRITTEN',
    'POSIX_SLOWEST_RANK_BYTES', 'POSIX_F_SLOWEST_RANK_TIME',
    'POSIX_F_MAX_READ_TIME', 'POSIX_F_MAX_WRITE_TIME',
    'POSIX_F_VARIANCE_RANK_TIME', 'POSIX_F_VARIANCE_RANK_BYTES',
    'POSIX_F_CLOSE_END_TIMESTAMP',
    'POSIX_F_READ_END_TIMESTAMP', 'POSIX_F_WRITE_END_TIMESTAMP',
    'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
    'POSIX_MAX_READ_TIME_SIZE', 'POSIX_MAX_WRITE_TIME_SIZE',
    'MPIIO_F_VARIANCE_RANK_TIME', 'MPIIO_F_VARIANCE_RANK_BYTES',
    'STDIO_F_VARIANCE_RANK_TIME', 'STDIO_F_VARIANCE_RANK_BYTES',
}

# Counters that should be aggregated with MIN
MIN_COUNTERS = {
    'POSIX_FASTEST_RANK_BYTES', 'POSIX_F_FASTEST_RANK_TIME',
    'POSIX_F_OPEN_START_TIMESTAMP',
    'POSIX_F_READ_START_TIMESTAMP', 'POSIX_F_WRITE_START_TIMESTAMP',
}

# Counters to skip (rank IDs, modes — not meaningful to sum)
SKIP_COUNTERS = {
    'POSIX_FASTEST_RANK', 'POSIX_SLOWEST_RANK',
    'POSIX_MODE', 'POSIX_RENAMED_FROM',
    'MPIIO_MODE',
    'STDIO_FASTEST_RANK', 'STDIO_SLOWEST_RANK',
    'MPIIO_FASTEST_RANK', 'MPIIO_SLOWEST_RANK',
}

# Special counters: stride and access patterns use "most common" logic
# For simplicity, when aggregating across files, we take the values from
# the file with the most I/O (largest total bytes)
ACCESS_PATTERN_COUNTERS = {
    'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE',
    'POSIX_STRIDE3_STRIDE', 'POSIX_STRIDE4_STRIDE',
    'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
    'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT',
    'POSIX_ACCESS1_ACCESS', 'POSIX_ACCESS2_ACCESS',
    'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
    'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT',
    'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT',
}


def aggregate_file_records(file_records):
    """Aggregate a list of per-file counter dicts into one job-level dict.

    Parameters
    ----------
    file_records : list[dict]
        Each dict maps counter names to numeric values for one file record.

    Returns
    -------
    dict
        Aggregated counters for the entire job.
    """
    if not file_records:
        return {}

    if len(file_records) == 1:
        result = dict(file_records[0])
        result['num_files'] = 1
        return result

    # Find the dominant file (highest total bytes) for access pattern counters
    dominant_idx = 0
    max_bytes = 0
    for i, rec in enumerate(file_records):
        total = rec.get('POSIX_BYTES_READ', 0) + rec.get('POSIX_BYTES_WRITTEN', 0)
        if total > max_bytes:
            max_bytes = total
            dominant_idx = i

    # Collect all counter names
    all_keys = set()
    for rec in file_records:
        all_keys.update(rec.keys())

    result = {}
    for key in all_keys:
        if key in SKIP_COUNTERS:
            continue

        if key in ACCESS_PATTERN_COUNTERS:
            # Take from dominant file
            result[key] = file_records[dominant_idx].get(key, 0)
            continue

        values = [rec.get(key, 0) for rec in file_records]
        # Replace sentinels (-1) with 0 for aggregation
        values = [v if v != -1 else 0 for v in values]

        if key in MAX_COUNTERS:
            result[key] = max(values)
        elif key in MIN_COUNTERS:
            positive = [v for v in values if v > 0]
            result[key] = min(positive) if positive else 0
        else:
            # Default: sum
            result[key] = sum(values)

    result['num_files'] = len(file_records)
    return result


def aggregate_from_total_output(counters):
    """Pass through counters from ``darshan-parser --total`` (already aggregated).

    When using ``darshan-parser --total``, the output already provides
    job-level aggregation. This function just cleans sentinel values and
    adds ``num_files`` if missing.

    Parameters
    ----------
    counters : dict
        Counter dictionary from CLI parser.

    Returns
    -------
    dict
        Cleaned counter dictionary.
    """
    result = {}
    for key, val in counters.items():
        if key in SKIP_COUNTERS:
            continue
        if val == -1:
            val = 0
        result[key] = val

    result.setdefault('num_files', 0)
    return result
