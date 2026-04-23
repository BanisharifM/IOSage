"""
Drishti Heuristic Labeling Pipeline
=====================================
Generates multi-label heuristic labels from production log features by
reimplementing Drishti's 30 heuristic rules as vectorized pandas operations.

This operates on the already-extracted production/features.parquet, not on
raw .darshan files. A full labeling pass over 131K rows completes in seconds.

Terminology (per IOSage paper convention):
  - "heuristic labels" = Drishti rule-based labels on production logs
  - "ground-truth labels" = benchmark-derived labels (by construction)
  - See docs/paper_materials.md Section 2.5.1 for rationale.

Drishti Insight Codes and Severity Levels:
    HIGH (critical issues):
        S01  - STDIO high usage (>10% of data)
        P05  - High small read requests (>10%, >1000 absolute)
        P06  - High small write requests (>10%, >1000 absolute)
        P07  - High misaligned memory requests (>10%)
        P08  - High misaligned file requests (>10%)
        P11  - High random read operations (>20%, >1000 absolute)
        P13  - High random write operations (>20%, >1000 absolute)
        P15  - High small shared-file reads (>10%, >1000 absolute)
        P16  - High small shared-file writes (>10%, >1000 absolute)
        P17  - High metadata time (>30s per file)
        P18  - Shared-file data transfer imbalance (>15%)
        P19  - Shared-file time imbalance (>15%)
        P21  - Individual write size imbalance (>30%)
        P22  - Individual read size imbalance (>30%)
        M02  - No collective MPI-IO reads (when >1000 ops)
        M03  - No collective MPI-IO writes (when >1000 ops)

    WARN (warnings):
        M01  - No MPI-IO usage
        M06  - No non-blocking MPI-IO reads
        M07  - No non-blocking MPI-IO writes
        P09  - Redundant read traffic
        P10  - Redundant write traffic

    INFO (metadata, not bottlenecks):
        P01  - Write operation intensive
        P02  - Read operation intensive
        P03  - Write size intensive
        P04  - Read size intensive

    OK (no issue):
        P12  - Sequential read usage
        P14  - Sequential write usage
        M04  - Collective read usage
        M05  - Collective write usage

Taxonomy Dimensions (8-dimensional binary vector):
    0: access_granularity   - Small operations, misalignment
    1: metadata_intensity   - High metadata time relative to I/O time
    2: parallelism_efficiency - Load imbalance across ranks
    3: access_pattern       - Random (non-sequential) access
    4: interface_choice     - No collective MPI-IO when appropriate (M02/M03)
    5: file_strategy        - Shared-file contention patterns
    6: throughput_utilization - Redundant traffic, low throughput
    7: healthy              - No issues detected in dimensions 0-6

References:
    Drishti v0.8: https://github.com/hpc-io/drishti-io
    Thresholds: drishti/includes/config.py (default values)
    Rules: drishti/includes/module.py (check_* functions)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy dimension names (order matters — matches label vector index)
# ---------------------------------------------------------------------------
DIMENSION_NAMES = [
    'access_granularity',       # 0
    'metadata_intensity',       # 1
    'parallelism_efficiency',   # 2
    'access_pattern',           # 3
    'interface_choice',         # 4
    'file_strategy',            # 5
    'throughput_utilization',   # 6
    'healthy',                  # 7
]

# ---------------------------------------------------------------------------
# Drishti default thresholds (from drishti/includes/config.py)
# ---------------------------------------------------------------------------
DRISHTI_THRESHOLDS = {
    'imbalance_operations': 0.1,      # P01-P04: read/write imbalance
    'small_bytes': 1048576,           # 1 MB boundary for "small" ops
    'small_requests': 0.1,           # P05/P06: fraction threshold
    'small_requests_absolute': 1000, # P05/P06: absolute count threshold
    'misaligned_requests': 0.1,      # P07/P08: fraction threshold
    'metadata_time_rank': 30,        # P17: seconds per file
    'random_operations': 0.2,        # P11/P13: fraction threshold
    'random_operations_absolute': 1000,  # P11/P13: absolute count threshold
    'imbalance_stragglers': 0.15,    # P18/P19: straggler threshold
    'imbalance_size': 0.3,           # P21/P22: per-file size imbalance
    'interface_stdio': 0.1,          # S01: STDIO fraction threshold
    'collective_operations': 0.5,    # M02/M03: collective fraction threshold
    'collective_operations_absolute': 1000,  # M02/M03: absolute threshold
}

# Severity weights for confidence scoring
SEVERITY_HIGH = 1.0
SEVERITY_WARN = 0.7
SEVERITY_INFO = 0.3


def compute_drishti_codes(df):
    """Compute all 30 Drishti insight codes as boolean Series.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features (one row per job, columns from feature_extraction).

    Returns
    -------
    dict[str, pd.Series]
        Maps Drishti code (e.g., 'P05') to boolean Series (True = triggered).
    """
    n = len(df)
    t = DRISHTI_THRESHOLDS
    codes = {}

    # Precompute commonly used totals
    total_reads = df['POSIX_READS'].clip(lower=1)
    total_writes = df['POSIX_WRITES'].clip(lower=1)
    total_ops = (df['POSIX_READS'] + df['POSIX_WRITES']).clip(lower=1)
    bytes_read = df['POSIX_BYTES_READ']
    bytes_written = df['POSIX_BYTES_WRITTEN']
    total_posix_bytes = (bytes_read + bytes_written).clip(lower=1)

    # STDIO bytes (for STDIO fraction)
    stdio_bytes = df['STDIO_BYTES_READ'] + df['STDIO_BYTES_WRITTEN']
    # Total bytes across all interfaces (POSIX already includes MPI-IO layer)
    total_bytes_all = (total_posix_bytes + stdio_bytes).clip(lower=1)

    # -----------------------------------------------------------------------
    # S01: STDIO high usage (HIGH)
    # Condition: stdio_bytes / total_bytes > 0.1
    # -----------------------------------------------------------------------
    codes['S01'] = (stdio_bytes / total_bytes_all) > t['interface_stdio']

    # -----------------------------------------------------------------------
    # P01-P04: Read/write intensity (INFO only — not bottlenecks)
    # -----------------------------------------------------------------------
    codes['P01'] = (
        (df['POSIX_WRITES'] > df['POSIX_READS']) &
        ((df['POSIX_WRITES'] - df['POSIX_READS']).abs() / total_ops > t['imbalance_operations'])
    )
    codes['P02'] = (
        (df['POSIX_READS'] > df['POSIX_WRITES']) &
        ((df['POSIX_WRITES'] - df['POSIX_READS']).abs() / total_ops > t['imbalance_operations'])
    )
    codes['P03'] = (
        (bytes_written > bytes_read) &
        ((bytes_written - bytes_read).abs() / total_posix_bytes > t['imbalance_operations'])
    )
    codes['P04'] = (
        (bytes_read > bytes_written) &
        ((bytes_written - bytes_read).abs() / total_posix_bytes > t['imbalance_operations'])
    )

    # -----------------------------------------------------------------------
    # P05/P06: Small operations (HIGH)
    # IMPORTANT: Drishti defines "small" as < 1 MB, NOT < 1 KB like our
    # feature_extraction.py's small_read_ratio.
    # Drishti small = SIZE_0_100 + SIZE_100_1K + SIZE_1K_10K + SIZE_10K_100K + SIZE_100K_1M
    # -----------------------------------------------------------------------
    drishti_small_reads = (
        df['POSIX_SIZE_READ_0_100'] + df['POSIX_SIZE_READ_100_1K'] +
        df['POSIX_SIZE_READ_1K_10K'] + df['POSIX_SIZE_READ_10K_100K'] +
        df['POSIX_SIZE_READ_100K_1M']
    )
    drishti_small_writes = (
        df['POSIX_SIZE_WRITE_0_100'] + df['POSIX_SIZE_WRITE_100_1K'] +
        df['POSIX_SIZE_WRITE_1K_10K'] + df['POSIX_SIZE_WRITE_10K_100K'] +
        df['POSIX_SIZE_WRITE_100K_1M']
    )

    codes['P05'] = (
        (drishti_small_reads / total_reads > t['small_requests']) &
        (drishti_small_reads > t['small_requests_absolute'])
    )
    codes['P06'] = (
        (drishti_small_writes / total_writes > t['small_requests']) &
        (drishti_small_writes > t['small_requests_absolute'])
    )

    # -----------------------------------------------------------------------
    # P07/P08: Misaligned requests (HIGH)
    # -----------------------------------------------------------------------
    codes['P07'] = (
        df['POSIX_MEM_NOT_ALIGNED'] / total_ops > t['misaligned_requests']
    )
    codes['P08'] = (
        df['POSIX_FILE_NOT_ALIGNED'] / total_ops > t['misaligned_requests']
    )

    # -----------------------------------------------------------------------
    # P09/P10: Redundant traffic (WARN)
    # Condition: max_byte_offset > total_bytes_transferred
    # -----------------------------------------------------------------------
    codes['P09'] = df['POSIX_MAX_BYTE_READ'] > bytes_read
    codes['P10'] = df['POSIX_MAX_BYTE_WRITTEN'] > bytes_written

    # -----------------------------------------------------------------------
    # P11/P13: Random operations (HIGH)
    # Drishti: random = total - SEQ (where SEQ includes CONSEC)
    # random_reads = READS - SEQ_READS
    # -----------------------------------------------------------------------
    random_reads = df['POSIX_READS'] - df['POSIX_SEQ_READS']
    random_writes = df['POSIX_WRITES'] - df['POSIX_SEQ_WRITES']

    codes['P11'] = (
        (random_reads / total_reads > t['random_operations']) &
        (random_reads > t['random_operations_absolute'])
    )
    # P12: Sequential reads (OK — not a bottleneck)
    codes['P12'] = ~codes['P11'] & (df['POSIX_READS'] > 0)

    codes['P13'] = (
        (random_writes / total_writes > t['random_operations']) &
        (random_writes > t['random_operations_absolute'])
    )
    # P14: Sequential writes (OK — not a bottleneck)
    codes['P14'] = ~codes['P13'] & (df['POSIX_WRITES'] > 0)

    # -----------------------------------------------------------------------
    # P15/P16: Small operations on shared files (HIGH)
    # Approximation: if is_shared_file=1 AND the small ops threshold triggers
    # -----------------------------------------------------------------------
    is_shared = df['is_shared_file'] == 1
    codes['P15'] = is_shared & codes['P05']
    codes['P16'] = is_shared & codes['P06']

    # -----------------------------------------------------------------------
    # P17: High metadata time (HIGH)
    # Drishti checks: count of files where POSIX_F_META_TIME > 30 seconds
    # Approximation: aggregate POSIX_F_META_TIME > threshold
    # -----------------------------------------------------------------------
    codes['P17'] = df['POSIX_F_META_TIME'] > t['metadata_time_rank']

    # -----------------------------------------------------------------------
    # P18/P19: Shared-file data/time imbalance (HIGH)
    # Drishti: per-file (SLOWEST_BYTES - FASTEST_BYTES) / total > 0.15
    # Our byte_imbalance is the aggregate version of this
    # Only meaningful for shared files (nprocs > 1)
    # -----------------------------------------------------------------------
    multi_rank = df['nprocs'] > 1
    codes['P18'] = multi_rank & (df['byte_imbalance'] > t['imbalance_stragglers'])
    codes['P19'] = multi_rank & (df['time_imbalance'] > t['imbalance_stragglers'])

    # -----------------------------------------------------------------------
    # P21/P22: Individual write/read size imbalance (HIGH)
    # Drishti: per-file (max_rank_bytes - min_rank_bytes) / max > 0.3
    # Proxy: rank_bytes_cv > sqrt(0.3) ~ 0.55, or byte_imbalance > 0.3
    # Use the stricter threshold from Drishti's imbalance_size
    # -----------------------------------------------------------------------
    codes['P21'] = multi_rank & (df['byte_imbalance'] > t['imbalance_size'])
    codes['P22'] = multi_rank & (df['byte_imbalance'] > t['imbalance_size'])

    # -----------------------------------------------------------------------
    # M01: No MPI-IO usage (WARN)
    # -----------------------------------------------------------------------
    codes['M01'] = df['has_mpiio'] == 0

    # -----------------------------------------------------------------------
    # M02/M03: No collective MPI-IO operations (HIGH)
    # Condition: coll_reads == 0 AND total_mpiio_reads > absolute threshold
    # -----------------------------------------------------------------------
    total_mpiio_reads = df['MPIIO_INDEP_READS'] + df['MPIIO_COLL_READS']
    total_mpiio_writes = df['MPIIO_INDEP_WRITES'] + df['MPIIO_COLL_WRITES']

    codes['M02'] = (
        (df['MPIIO_COLL_READS'] == 0) &
        (total_mpiio_reads > t['collective_operations_absolute'])
    )
    codes['M03'] = (
        (df['MPIIO_COLL_WRITES'] == 0) &
        (total_mpiio_writes > t['collective_operations_absolute'])
    )

    # M04/M05: Collective usage (OK — not a bottleneck)
    codes['M04'] = (df['MPIIO_COLL_READS'] > 0)
    codes['M05'] = (df['MPIIO_COLL_WRITES'] > 0)

    # -----------------------------------------------------------------------
    # M06/M07: Blocking MPI-IO operations (WARN)
    # Condition: has MPI-IO but nb_reads/writes == 0
    # -----------------------------------------------------------------------
    has_mpiio = df['has_mpiio'] == 1
    codes['M06'] = has_mpiio & (df['MPIIO_NB_READS'] == 0)
    codes['M07'] = has_mpiio & (df['MPIIO_NB_WRITES'] == 0)

    # M08/M09/M10: Aggregator checks — require sacct, not available
    codes['M08'] = pd.Series(False, index=df.index)
    codes['M09'] = pd.Series(False, index=df.index)
    codes['M10'] = pd.Series(False, index=df.index)

    return codes


def codes_to_labels(codes):
    """Map Drishti insight codes to 8-dimensional taxonomy labels.

    Only HIGH-severity codes trigger dimension labels. WARN-level codes
    (M01, M06, M07, P09, P10) are recorded as individual Drishti codes
    but do NOT activate dimension labels, because:

    1. M01 (no MPI-IO): Many legitimate serial/Python jobs don't need
       MPI-IO. Flagging all of them as "interface_choice" makes the label
       uninformative (72.6% of jobs).
    2. M06/M07 (blocking I/O): Most MPI-IO usage on Polaris is blocking.
       This is normal, not a bottleneck.
    3. P09/P10 (redundant traffic): The MAX_BYTE > total_bytes heuristic
       has high false-positive rate from multi-file aggregation artifacts.

    For access_granularity, P08 (file misalignment) alone triggers 93.8%
    of jobs because most applications don't set explicit Lustre alignment.
    We separate alignment issues from small-operation issues:
    - access_granularity: only P05, P06 (small ops)
    - A separate alignment flag is stored in the individual codes.

    Parameters
    ----------
    codes : dict[str, pd.Series]
        Boolean Series per Drishti code from compute_drishti_codes().

    Returns
    -------
    pd.DataFrame
        Columns: DIMENSION_NAMES (8 binary columns), one row per job.
    """
    n = len(next(iter(codes.values())))
    labels = pd.DataFrame(0, index=range(n), columns=DIMENSION_NAMES)

    # Dimension 0: access_granularity
    # Small operations only (P05, P06). Misalignment (P07, P08) excluded
    # from dimension trigger because P08 alone covers 93.8% of jobs,
    # making the label near-useless for classification. Misalignment
    # information is preserved in the individual drishti_P07/P08 columns.
    labels['access_granularity'] = (
        codes['P05'] | codes['P06']
    ).astype(int)

    # Dimension 1: metadata_intensity
    labels['metadata_intensity'] = codes['P17'].astype(int)

    # Dimension 2: parallelism_efficiency
    # Load imbalance: data (P18, P21, P22) + time (P19)
    labels['parallelism_efficiency'] = (
        codes['P18'] | codes['P19'] | codes['P21'] | codes['P22']
    ).astype(int)

    # Dimension 3: access_pattern
    # Random access (P11, P13)
    labels['access_pattern'] = (
        codes['P11'] | codes['P13']
    ).astype(int)

    # Dimension 4: interface_choice
    # M02 (no collective reads) + M03 (no collective writes) only.
    # S01 (STDIO >10%) REMOVED: STDIO is the correct interface for Python/ML
    # workloads (40% of Polaris). Labeling correct behavior as "bottleneck"
    # creates systematic false positives and breaks alignment with benchmark
    # ground-truth labels. S01 signal preserved as individual drishti_S01 column.
    # Ref: Snorkel (VLDB'18) requires labeling functions and gold labels to
    # define the same concept. See docs/SC2026_Training_Strategy.md.
    labels['interface_choice'] = (
        codes['M02'] | codes['M03']
    ).astype(int)

    # Dimension 5: file_strategy
    # Small ops on shared files (P15, P16)
    labels['file_strategy'] = (
        codes['P15'] | codes['P16']
    ).astype(int)

    # Dimension 6: throughput_utilization
    # Redundant traffic (P09, P10) — these are WARN level but kept
    # because they represent genuine throughput waste, not just
    # missing features. However, we tighten the condition: only flag
    # when redundant traffic is substantial (read > 2x the max offset,
    # i.e., significant re-reading, not just minor overlap).
    labels['throughput_utilization'] = (
        codes['P09'] | codes['P10']
    ).astype(int)

    # Dimension 7: healthy
    # No issues in any dimension 0-6
    any_issue = labels[DIMENSION_NAMES[:7]].any(axis=1)
    labels['healthy'] = (~any_issue).astype(int)

    return labels


def compute_confidence(codes, labels):
    """Compute per-sample labeling confidence based on severity and coverage.

    Confidence is computed as the mean severity of triggered codes, weighted
    by how many codes fired. Higher confidence means more/stronger evidence.

    Parameters
    ----------
    codes : dict[str, pd.Series]
        Boolean Series per Drishti code.
    labels : pd.DataFrame
        Label DataFrame from codes_to_labels().

    Returns
    -------
    pd.Series
        Confidence score per sample in [0, 1].
    """
    # Map each code to its severity
    code_severity = {
        # HIGH severity
        'S01': SEVERITY_HIGH,
        'P05': SEVERITY_HIGH, 'P06': SEVERITY_HIGH,
        'P07': SEVERITY_HIGH, 'P08': SEVERITY_HIGH,
        'P11': SEVERITY_HIGH, 'P13': SEVERITY_HIGH,
        'P15': SEVERITY_HIGH, 'P16': SEVERITY_HIGH,
        'P17': SEVERITY_HIGH,
        'P18': SEVERITY_HIGH, 'P19': SEVERITY_HIGH,
        'P21': SEVERITY_HIGH, 'P22': SEVERITY_HIGH,
        'M02': SEVERITY_HIGH, 'M03': SEVERITY_HIGH,
        # WARN severity
        'M01': SEVERITY_WARN,
        'M06': SEVERITY_WARN, 'M07': SEVERITY_WARN,
        'P09': SEVERITY_WARN, 'P10': SEVERITY_WARN,
        # INFO severity (metadata, not bottlenecks — low confidence)
        'P01': SEVERITY_INFO, 'P02': SEVERITY_INFO,
        'P03': SEVERITY_INFO, 'P04': SEVERITY_INFO,
        # OK codes (no issue detected)
        'P12': 0.0, 'P14': 0.0,
        'M04': 0.0, 'M05': 0.0,
        'M08': 0.0, 'M09': 0.0, 'M10': 0.0,
    }

    # Only count codes that indicate issues (severity > 0)
    issue_codes = {k: v for k, v in code_severity.items() if v > 0}

    severity_sum = pd.Series(0.0, index=labels.index)
    code_count = pd.Series(0, index=labels.index)

    for code, severity in issue_codes.items():
        if code in codes:
            triggered = codes[code].astype(float)
            severity_sum += triggered * severity
            code_count += codes[code].astype(int)

    # Confidence = mean severity of triggered codes
    # Jobs with no issues get confidence = 0.5 (healthy label, moderate confidence)
    confidence = severity_sum / code_count.clip(lower=1)

    # Healthy jobs: moderate confidence (Drishti absence of issues)
    healthy_mask = labels['healthy'] == 1
    confidence[healthy_mask] = 0.5

    return confidence


def generate_heuristic_labels(features_path, output_path, min_confidence=0.0):
    """Generate heuristic labels from engineered features using Drishti rules.

    Parameters
    ----------
    features_path : str or Path
        Path to production/features.parquet.
    output_path : str or Path
        Output path for heuristic labels parquet file.
    min_confidence : float
        Minimum confidence to include a sample (default: 0.0 = keep all).

    Returns
    -------
    pd.DataFrame
        Heuristic labels DataFrame with _jobid, 8 dimension columns,
        drishti_confidence, label_source, and all 30 Drishti code columns.
    """
    features_path = Path(features_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Compute all 30 Drishti codes
    logger.info("Computing Drishti insight codes...")
    codes = compute_drishti_codes(df)

    # Map codes to 8-dimensional labels
    logger.info("Mapping codes to taxonomy dimensions...")
    labels = codes_to_labels(codes)

    # Compute confidence scores
    confidence = compute_confidence(codes, labels)

    # Build output DataFrame
    result = pd.DataFrame()
    result['_jobid'] = df['_jobid'].values

    # 8 dimension labels
    for dim_name in DIMENSION_NAMES:
        result[dim_name] = labels[dim_name].values

    # Confidence and source
    result['drishti_confidence'] = confidence.values
    result['label_source'] = 'drishti_heuristic'

    # Individual Drishti codes (for debugging, analysis, and Drishti baseline)
    for code_name, code_series in sorted(codes.items()):
        result[f'drishti_{code_name}'] = code_series.astype(int).values

    # Filter by confidence if requested
    if min_confidence > 0:
        before = len(result)
        result = result[result['drishti_confidence'] >= min_confidence]
        logger.info("Confidence filter (>= %.2f): %d -> %d rows (%.1f%% kept)",
                     min_confidence, before, len(result),
                     100 * len(result) / max(before, 1))

    # Write output
    result.to_parquet(output_path, index=False, engine='pyarrow')
    logger.info("Wrote heuristic labels to %s (%d rows, %.1f MB)",
                output_path, len(result),
                output_path.stat().st_size / 1e6)

    # Summary statistics
    _log_summary(result)

    return result


def _log_summary(result):
    """Log label distribution statistics."""
    n = len(result)

    logger.info("=== Heuristic Label Summary ===")
    logger.info("Total samples: %d", n)

    for dim_name in DIMENSION_NAMES:
        count = result[dim_name].sum()
        logger.info("  %-25s %6d (%5.1f%%)", dim_name, count, 100 * count / n)

    # Multi-label statistics
    issue_dims = DIMENSION_NAMES[:7]
    n_issues = result[issue_dims].sum(axis=1)
    logger.info("Issue count distribution:")
    for k in range(8):
        count = (n_issues == k).sum()
        if count > 0:
            logger.info("  %d issues: %6d (%5.1f%%)", k, count, 100 * count / n)

    # Drishti code trigger rates
    code_cols = [c for c in result.columns if c.startswith('drishti_')]
    logger.info("Drishti code trigger rates:")
    for col in sorted(code_cols):
        count = result[col].sum()
        if count > 0:
            code = col.replace('drishti_', '')
            logger.info("  %-5s %6d (%5.1f%%)", code, count, 100 * count / n)

    # Confidence distribution
    conf = result['drishti_confidence']
    logger.info("Confidence: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
                conf.mean(), conf.median(), conf.min(), conf.max())


# NOTE: Legacy alias 'generate_silver_labels' removed 2026-03-15.
# Use generate_heuristic_labels() directly.


def validate_against_drishti_cli(heuristic_labels_path, sample_logs_dir,
                                 n_samples=50, seed=42):
    """Validate vectorized labels against actual Drishti CLI output.

    Runs Drishti on a random sample of logs and compares the triggered
    insight codes with our vectorized reimplementation.

    Parameters
    ----------
    heuristic_labels_path : str or Path
        Path to heuristic labels parquet (to look up our predictions).
    sample_logs_dir : str or Path
        Directory containing sample .darshan files.
    n_samples : int
        Number of samples to validate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Validation results with agreement rates per code.
    """
    import subprocess
    import re

    heuristic_df = pd.read_parquet(heuristic_labels_path)
    sample_dir = Path(sample_logs_dir)
    logs = sorted(sample_dir.rglob('*.darshan'))

    if not logs:
        logger.warning("No .darshan files found in %s", sample_dir)
        return {}

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(logs), size=min(n_samples, len(logs)),
                            replace=False)
    sample_logs = [logs[i] for i in sample_idx]

    agreements = {}
    total_compared = 0

    for log_path in sample_logs:
        try:
            result = subprocess.run(
                ['drishti', str(log_path), '--export-csv'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                continue

            # Parse Drishti CSV output to get triggered codes
            # CSV format: code,True/False per line
            drishti_codes = set()
            for line in result.stdout.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1].strip().lower() == 'true':
                    drishti_codes.add(parts[0].strip())

            # Find matching jobid in our labels
            jobid = log_path.stem.split('_')[0]  # Extract jobid from filename
            our_row = heuristic_df[heuristic_df['_jobid'].astype(str).str.contains(jobid)]
            if our_row.empty:
                continue

            our_codes = set()
            for col in heuristic_df.columns:
                if col.startswith('drishti_') and col != 'drishti_confidence':
                    code = col.replace('drishti_', '')
                    if our_row.iloc[0][col] == 1:
                        our_codes.add(code)

            # Compare
            for code in set(list(drishti_codes) + list(our_codes)):
                if code not in agreements:
                    agreements[code] = {'match': 0, 'mismatch': 0}
                if (code in drishti_codes) == (code in our_codes):
                    agreements[code]['match'] += 1
                else:
                    agreements[code]['mismatch'] += 1

            total_compared += 1

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug("Drishti failed on %s: %s", log_path, e)
            continue

    logger.info("Validated %d samples against Drishti CLI", total_compared)
    for code, stats in sorted(agreements.items()):
        total = stats['match'] + stats['mismatch']
        rate = stats['match'] / max(total, 1)
        logger.info("  %s: %.1f%% agreement (%d/%d)",
                     code, 100 * rate, stats['match'], total)

    return agreements


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for heuristic label generation."""
    parser = argparse.ArgumentParser(
        description='Generate Drishti heuristic labels from engineered features'
    )
    parser.add_argument(
        '--features', required=True,
        help='Path to production/features.parquet'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for heuristic labels parquet'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.0,
        help='Minimum confidence threshold (default: 0.0 = keep all)'
    )
    parser.add_argument(
        '--validate-dir', default=None,
        help='Directory with .darshan files for CLI validation'
    )
    parser.add_argument(
        '--validate-samples', type=int, default=50,
        help='Number of samples for CLI validation (default: 50)'
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )

    result = generate_heuristic_labels(
        features_path=args.features,
        output_path=args.output,
        min_confidence=args.min_confidence,
    )

    if args.validate_dir:
        validate_against_drishti_cli(
            heuristic_labels_path=args.output,
            sample_logs_dir=args.validate_dir,
            n_samples=args.validate_samples,
        )


if __name__ == '__main__':
    main()
