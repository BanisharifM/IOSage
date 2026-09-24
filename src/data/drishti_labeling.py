"""
Production Labeling and Drishti Diagnostic Pipeline
====================================================
Generates production labels with the shared IOSage rule contract and keeps
Drishti diagnostic codes for a separate baseline and diagnostic analysis.

This operates on engineered production features, not on raw Darshan logs.

Terminology (per IOSage paper convention):
  - "heuristic labels" = shared IOSage rules applied to production logs
  - "Drishti baseline" = taxonomy labels mapped from Drishti insight codes
  - "ground-truth labels" = benchmark-derived labels (by construction)
  - See docs/1_strategy/paper_materials.md Section 2.5.1 for rationale.

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
    0: access_granularity   - Small operations
    1: metadata_intensity   - High metadata time relative to I/O time
    2: parallelism_efficiency - Load imbalance across ranks
    3: access_pattern       - Random (non-sequential) access
    4: interface_choice     - Low collective use or POSIX shared-file access
    5: file_strategy        - Data files at least equal to process count
    6: throughput_utilization - Excessive synchronous writes
    7: healthy              - No issues detected in dimensions 0-6

References:
    Drishti v0.8: https://github.com/hpc-io/drishti-io
    Thresholds: drishti/includes/config.py (default values)
    Rules: drishti/includes/module.py (check_* functions)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.data.label_rules import (
    DIMENSION_NAMES,
    labels_from_features,
    validity_from_features,
)
from src.utils.artifacts import sha256_file, write_atomic

logger = logging.getLogger(__name__)

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
    """Compute 32 Drishti insight-code columns as boolean Series.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features (one row per job, columns from feature_extraction).

    Returns
    -------
    dict[str, pd.Series]
        Maps Drishti code (e.g., 'P05') to boolean Series (True = triggered).
    """
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
    # P01-P04: Read/write intensity (INFO only, not bottlenecks)
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
    # P12: Sequential reads (OK, not a bottleneck)
    codes['P12'] = ~codes['P11'] & (df['POSIX_READS'] > 0)

    codes['P13'] = (
        (random_writes / total_writes > t['random_operations']) &
        (random_writes > t['random_operations_absolute'])
    )
    # P14: Sequential writes (OK, not a bottleneck)
    codes['P14'] = ~codes['P13'] & (df['POSIX_WRITES'] > 0)

    # -----------------------------------------------------------------------
    # P15/P16: Small operations on shared files (HIGH)
    # The parser aggregates the request counts from shared POSIX records only.
    # -----------------------------------------------------------------------
    codes['P15'] = (
        (df['SHARED_POSIX_SMALL_READS'] > t['small_requests_absolute'])
        & (df['SHARED_POSIX_SMALL_READS']
           / df['SHARED_POSIX_READS'].clip(lower=1) > t['small_requests'])
    )
    codes['P16'] = (
        (df['SHARED_POSIX_SMALL_WRITES'] > t['small_requests_absolute'])
        & (df['SHARED_POSIX_SMALL_WRITES']
           / df['SHARED_POSIX_WRITES'].clip(lower=1) > t['small_requests'])
    )

    # -----------------------------------------------------------------------
    # P17: High metadata time (HIGH)
    # Drishti checks: count of files where POSIX_F_META_TIME > 30 seconds
    # Approximation: aggregate POSIX_F_META_TIME > threshold
    # -----------------------------------------------------------------------
    codes['P17'] = df['POSIX_F_META_TIME'] > t['metadata_time_rank']

    # -----------------------------------------------------------------------
    # P18/P19: Shared-file data/time imbalance (HIGH)
    # Drishti: per shared record, |SLOWEST_RANK_BYTES - FASTEST_RANK_BYTES| /
    # record bytes > 0.15 (P18) and the same on time (P19).
    # SHARED_BYTE_IMBALANCE / SHARED_TIME_IMBALANCE hold the largest such
    # value over the job's shared records (parse_darshan._shared_record_imbalance).
    # -----------------------------------------------------------------------
    multi_rank = df['nprocs'] > 1
    codes['P18'] = multi_rank & (df['SHARED_BYTE_IMBALANCE'] > t['imbalance_stragglers'])
    codes['P19'] = multi_rank & (df['SHARED_TIME_IMBALANCE'] > t['imbalance_stragglers'])

    # -----------------------------------------------------------------------
    # P21/P22: Individual write/read size imbalance (HIGH)
    # Drishti: per file, (max_rank_bytes - min_rank_bytes) / max > 0.3 over
    # the ranks that accessed that file (records with rank != -1).
    # FILE_WRITE_IMBALANCE / FILE_READ_IMBALANCE hold the largest such value
    # of the job (parse_darshan._per_file_imbalance).
    # -----------------------------------------------------------------------
    codes['P21'] = multi_rank & (df['FILE_WRITE_IMBALANCE'] > t['imbalance_size'])
    codes['P22'] = multi_rank & (df['FILE_READ_IMBALANCE'] > t['imbalance_size'])

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

    # M04/M05: Collective usage (OK, not a bottleneck)
    codes['M04'] = (df['MPIIO_COLL_READS'] > 0)
    codes['M05'] = (df['MPIIO_COLL_WRITES'] > 0)

    # -----------------------------------------------------------------------
    # M06/M07: Blocking MPI-IO operations (WARN)
    # Condition: has MPI-IO but nb_reads/writes == 0
    # -----------------------------------------------------------------------
    has_mpiio = df['has_mpiio'] == 1
    codes['M06'] = has_mpiio & (df['MPIIO_NB_READS'] == 0)
    codes['M07'] = has_mpiio & (df['MPIIO_NB_WRITES'] == 0)

    # M08/M09/M10: Aggregator checks require sacct, which is not available
    codes['M08'] = pd.Series(False, index=df.index)
    codes['M09'] = pd.Series(False, index=df.index)
    codes['M10'] = pd.Series(False, index=df.index)

    return codes


def codes_to_labels(codes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Map Drishti insight codes to the eight taxonomy dimensions.

    Parameters
    ----------
    codes : dict[str, pd.Series]
        Boolean Series per Drishti code from compute_drishti_codes().
    Returns
    -------
    pd.DataFrame
        Columns: DIMENSION_NAMES (8 binary columns), one row per job.
    """
    if not codes:
        raise ValueError("codes cannot be empty")
    required = {
        'P05', 'P06', 'P09', 'P10', 'P11', 'P13', 'P15', 'P16', 'P17',
        'P18', 'P19', 'P21', 'P22', 'M02', 'M03',
    }
    missing = required - set(codes)
    if missing:
        raise ValueError(f"Drishti codes lack {sorted(missing)}")
    index = next(iter(codes.values())).index
    if any(not series.index.equals(index) for series in codes.values()):
        raise ValueError("Drishti code indices do not align")
    labels = pd.DataFrame(0, index=index, columns=DIMENSION_NAMES)
    labels['access_granularity'] = (codes['P05'] | codes['P06']).astype(int)
    labels['metadata_intensity'] = codes['P17'].astype(int)
    labels['parallelism_efficiency'] = (
        codes['P18'] | codes['P19'] | codes['P21'] | codes['P22']
    ).astype(int)
    labels['access_pattern'] = (codes['P11'] | codes['P13']).astype(int)
    labels['interface_choice'] = (codes['M02'] | codes['M03']).astype(int)
    labels['file_strategy'] = (codes['P15'] | codes['P16']).astype(int)
    labels['throughput_utilization'] = (codes['P09'] | codes['P10']).astype(int)
    labels['healthy'] = (~labels[DIMENSION_NAMES[:7]].any(axis=1)).astype(int)
    if labels.isna().any().any() or not labels.isin([0, 1]).all().all():
        raise AssertionError("labels must be binary and complete")
    return labels


def compute_confidence(
    codes: Mapping[str, pd.Series], labels: pd.DataFrame,
) -> pd.Series:
    """Compute per-sample labeling confidence based on severity and coverage.

    Confidence is computed as the mean severity of triggered codes, weighted
    by how many codes fired. Higher confidence means more/stronger evidence.

    Parameters
    ----------
    codes : dict[str, pd.Series]
        Boolean Series per Drishti code.
    labels : pd.DataFrame
        Drishti baseline labels from ``codes_to_labels``.

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
        # INFO severity (metadata, not bottlenecks, low confidence)
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


def generate_heuristic_labels(
    features_path: str | Path, output_path: str | Path,
) -> pd.DataFrame:
    """Generate shared-rule labels and Drishti diagnostics from features.

    Parameters
    ----------
    features_path : str or Path
        Path to production/features.parquet.
    output_path : str or Path
        Output path for heuristic labels parquet file.
    Returns
    -------
    pd.DataFrame
        Heuristic labels DataFrame with _jobid, 8 dimension columns,
        Drishti diagnostic confidence, label source, validity columns, and all
        32 Drishti code columns.
    """
    features_path = Path(features_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(output_path.name + '.manifest.json')
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to replace labels or manifest: {output_path}")

    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    if df.empty:
        raise ValueError("features contain no rows; nothing written")

    # Compute the Drishti diagnostic codes.
    logger.info("Computing Drishti insight codes...")
    codes = compute_drishti_codes(df)

    # Production and benchmark verification use the same observable rules.
    logger.info("Applying the shared taxonomy rules...")
    labels = labels_from_features(df)
    validity = validity_from_features(df)

    # Keep the Drishti baseline and its confidence as separate diagnostics.
    drishti_labels = codes_to_labels(codes)
    confidence = compute_confidence(codes, drishti_labels)

    # Build output DataFrame; _source_path is the unique sample id that the
    # trainer joins on (_jobid repeats: one SLURM job holds many launches)
    if '_source_path' not in df.columns:
        raise ValueError("features lack _source_path; labels could not be joined back")
    result = pd.DataFrame()
    result['_source_path'] = df['_source_path'].values
    result['_jobid'] = df['_jobid'].values

    # 8 dimension labels
    for dim_name in DIMENSION_NAMES:
        result[dim_name] = labels[dim_name].values
        result[f'valid_{dim_name}'] = validity[dim_name].values

    # Confidence and source
    result['drishti_confidence'] = confidence.values
    result['label_source'] = 'iosage_shared_rules'

    # Individual Drishti codes (for debugging, analysis, and Drishti baseline)
    for code_name, code_series in sorted(codes.items()):
        result[f'drishti_{code_name}'] = code_series.astype(int).values

    # Write output and provenance without replacing prior evidence.
    write_atomic(
        output_path,
        lambda path: result.to_parquet(path, index=False, engine='pyarrow'),
    )

    manifest = {
        'schema_version': 3,
        'status': 'passed',
        'method': 'iosage_shared_rules',
        'diagnostics': 'drishti_codes_and_confidence',
        'features': {'path': str(features_path.resolve()), 'sha256': sha256_file(features_path)},
        'labels': {'path': str(output_path.resolve()), 'sha256': sha256_file(output_path),
                   'rows': len(result), 'columns': len(result.columns)},
        'script': {'path': str(Path(__file__).resolve()),
                   'sha256': sha256_file(Path(__file__).resolve())},
    }
    def write_manifest(path):
        with path.open('x') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')

    write_atomic(manifest_path, write_manifest)
    logger.info("Wrote heuristic labels to %s (%d rows, %.1f MB)",
                output_path, len(result),
                output_path.stat().st_size / 1e6)
    logger.info("Label manifest: %s", manifest_path)

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
    code_cols = [c for c in result.columns
                 if c.startswith('drishti_') and c != 'drishti_confidence']
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for heuristic label generation."""
    parser = argparse.ArgumentParser(
        description='Generate shared-rule labels and Drishti diagnostics'
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
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )

    generate_heuristic_labels(
        features_path=args.features,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
