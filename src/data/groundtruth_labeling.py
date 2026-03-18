"""Assign ground-truth labels from benchmark configuration, verified by Darshan.

Ground-truth labels are assigned by CONSTRUCTION: the benchmark config defines
the label, NOT Darshan feature values. This ensures non-circular ML evaluation
(SC reviewers can verify: IOR -t 64 IS small I/O by definition).

If verification fails (Darshan doesn't show the expected pattern), the log is
relabeled based on what Darshan actually shows (Option B from PROJECT_PHASES.md).

Usage:
    python -m src.data.groundtruth_labeling \
        --log-dir data/benchmark_logs \
        --output data/benchmark_logs/groundtruth_labels.csv

    python -m src.data.groundtruth_labeling \
        --log-dir data/benchmark_logs \
        --output data/processed/groundtruth_labels.parquet \
        --format parquet
"""

import argparse
import csv
import logging
import os
import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

from src.data.benchmark_verify import (
    DIMENSION_NAMES,
    verify_log_file,
    infer_labels_from_filename,
)

logger = logging.getLogger(__name__)

# Labels by construction: benchmark config → label
# Keys are glob patterns matching benchmark scenario names in filenames
BENCHMARK_LABEL_MAP = {
    # IOR small I/O (access_granularity bottleneck)
    'ior_small_posix_*': {'access_granularity': 1},
    'ior_small_direct_*': {'access_granularity': 1},
    'ior_misaligned_*': {'access_granularity': 1},
    # IOR random access
    'ior_random_posix_*': {'access_pattern': 1},
    'ior_random_small_*': {'access_pattern': 1, 'access_granularity': 1},
    # IOR interface misuse
    'ior_interface_posix_shared_*': {'interface_choice': 1},
    'ior_interface_mpiio_indep_*': {'interface_choice': 1},
    # IOR file explosion
    'ior_file_explosion_*': {'file_strategy': 1},
    # IOR fsync bottleneck
    'ior_fsync_per_write_*': {'throughput_utilization': 1},
    # IOR healthy
    'ior_healthy_*': {'healthy': 1},
    # mdtest metadata
    'mdtest_meta_*': {'metadata_intensity': 1},
    'mdtest_deep_tree_*': {'metadata_intensity': 1},
    # mdtest file explosion
    'mdtest_fpp_*': {'file_strategy': 1},
    # mdtest healthy
    'mdtest_healthy_*': {'healthy': 1},
    # DLIO small records
    'dlio_small_*': {'access_granularity': 1},
    # DLIO checkpoint
    'dlio_ckpt_*': {'throughput_utilization': 1},
    # DLIO healthy
    'dlio_healthy_*': {'healthy': 1},
    # DLIO shuffle
    'dlio_shuffle_*': {'access_pattern': 1},
    # Custom imbalance
    'custom_imbalance_*': {'parallelism_efficiency': 1},
    # Custom balanced
    'custom_balanced_*': {'healthy': 1},
    # IO500-style configs (used by ION paper for Drishti comparison)
    'ior_io500_hard_*': {'access_granularity': 1, 'interface_choice': 1},
    'ior_io500_easy_*': {'healthy': 1},
    'mdtest_io500_hard_*': {'metadata_intensity': 1},
    'mdtest_io500_easy_*': {'healthy': 1},
    # E2E-style API comparison
    'ior_e2e_posix_shared_*': {'interface_choice': 1},
    'ior_e2e_mpiio_coll_*': {'healthy': 1},
    # h5bench scenarios
    # Note: INTERLEAVED in h5bench = HDF5 compound datatype (AOS layout), NOT random
    # POSIX access. HDF5 internal pipeline converts to sequential POSIX writes.
    # Ref: Bez et al. (ACM CSUR 2023) on I/O stack abstraction gap.
    'h5b_indep_small_n*': {'interface_choice': 1},
    'h5b_interleaved_access_*': {'healthy': 1},
    'h5b_collective_small_*': {'access_granularity': 1},
    'h5b_collective_large_healthy_*': {'healthy': 1},
    'h5b_indep_large_healthy_*': {'healthy': 1},
    'h5b_indep_small_interleaved_*': {'access_granularity': 1, 'interface_choice': 1},
    'h5b_indep_interleaved_*': {'interface_choice': 1},
    'h5b_indep_small_single_ost_*': {'access_granularity': 1, 'interface_choice': 1, 'throughput_utilization': 1},
    # HACC-IO scenarios
    'hacc_posix_shared_large_*': {'interface_choice': 1},
    'hacc_fpp_many_ranks_*': {'file_strategy': 1},
    'hacc_posix_shared_single_ost_*': {'throughput_utilization': 1},
    'hacc_mpiio_collective_healthy_*': {'healthy': 1},
    'hacc_fpp_healthy_*': {'healthy': 1},
    'hacc_posix_shared_small_p*': {'access_granularity': 1, 'interface_choice': 1},
    'hacc_posix_shared_small_1ost_*': {'access_granularity': 1, 'interface_choice': 1, 'throughput_utilization': 1},
    'hacc_fpp_small_many_*': {'file_strategy': 1, 'access_granularity': 1},
    'hacc_posix_shared_many_single_ost_*': {'interface_choice': 1, 'throughput_utilization': 1},
}


def assign_groundtruth_label(log_filename, darshan_features=None):
    """Assign ground-truth label from benchmark configuration.

    Step 1: Match filename to BENCHMARK_LABEL_MAP (construction-based).
    Step 2: If darshan_features provided, verify pattern matches.
    Step 3: If verification fails, relabel from Darshan evidence.
    Step 4: Set healthy flag if all bottleneck dimensions are 0.

    Args:
        log_filename: Darshan log filename (basename)
        darshan_features: Optional dict of extracted features for verification

    Returns:
        (labels: dict, source: str, verified: bool)
        source is 'construction' or 'relabeled' or 'inferred'
    """
    labels = {dim: 0 for dim in DIMENSION_NAMES}
    source = 'construction'
    verified = True

    # Step 1: Construction-based label from filename pattern
    name_lower = log_filename.lower()
    matched = False
    for pattern, pattern_labels in BENCHMARK_LABEL_MAP.items():
        if fnmatch(name_lower, pattern):
            labels.update(pattern_labels)
            matched = True
            break

    if not matched:
        # Try inference from filename
        labels = infer_labels_from_filename(log_filename)
        source = 'inferred'
        if all(v == 0 for v in labels.values()):
            logger.warning(
                "Cannot determine label for %s (no matching pattern)", log_filename
            )

    # Step 2: Verify with Darshan features (if available)
    if darshan_features is not None and matched:
        from src.data.benchmark_verify import verify_benchmark_log
        passed, report = verify_benchmark_log(darshan_features, labels)
        verified = passed

        # Step 3: If verification fails, relabel from Darshan evidence
        if not passed:
            logger.warning(
                "Verification failed for %s, relabeling from Darshan", log_filename
            )
            labels = relabel_from_darshan(darshan_features)
            source = 'relabeled'

    # Step 4: Set healthy flag
    bottleneck_dims = [v for k, v in labels.items() if k != 'healthy']
    if sum(bottleneck_dims) == 0:
        labels['healthy'] = 1
    else:
        labels['healthy'] = 0

    return labels, source, verified


def relabel_from_darshan(features):
    """Assign labels based on what Darshan actually shows (fallback).

    Used when verification fails — the ML model sees features at inference,
    so labels must match what the features actually show.
    """
    labels = {dim: 0 for dim in DIMENSION_NAMES}

    # Access granularity
    small_io_ratio = features.get('small_io_ratio', 0)
    if small_io_ratio > 0.5:
        labels['access_granularity'] = 1

    # Metadata intensity
    meta_ratio = features.get('metadata_time_ratio', 0)
    if meta_ratio > 0.3:
        labels['metadata_intensity'] = 1

    # Parallelism efficiency
    rank_cv = features.get('rank_bytes_cv', 0)
    if rank_cv > 0.3:
        labels['parallelism_efficiency'] = 1

    # Access pattern
    seq_write = features.get('seq_write_ratio', 1)
    seq_read = features.get('seq_read_ratio', 1)
    if seq_write < 0.5 and seq_read < 0.5:
        labels['access_pattern'] = 1

    # Interface choice
    has_mpiio = features.get('has_mpiio', 0)
    coll_writes = features.get('MPIIO_COLL_WRITES', 0)
    nprocs = features.get('nprocs', 1)
    is_shared = features.get('is_shared_file', 0)
    if is_shared and nprocs > 4 and (has_mpiio == 0 or coll_writes == 0):
        labels['interface_choice'] = 1

    # File strategy
    num_files = features.get('num_files', 0)
    if num_files > 100:
        labels['file_strategy'] = 1

    # Throughput utilization
    write_bw = features.get('write_bw_mb_s', float('inf'))
    if write_bw < 10 and features.get('POSIX_BYTES_WRITTEN', 0) > 1e6:
        labels['throughput_utilization'] = 1

    # Healthy
    if sum(v for k, v in labels.items() if k != 'healthy') == 0:
        labels['healthy'] = 1

    return labels


def process_benchmark_logs(log_dir, output_path, output_format='csv'):
    """Process all benchmark Darshan logs and generate ground-truth labels.

    Args:
        log_dir: Root directory with benchmark .darshan files
        output_path: Path for output labels file
        output_format: 'csv' or 'parquet'

    Returns:
        DataFrame with labels and metadata
    """
    log_dir = Path(log_dir)
    darshan_files = sorted(log_dir.glob('**/*.darshan'))

    if not darshan_files:
        logger.error("No .darshan files found in %s", log_dir)
        return None

    logger.info("Processing %d benchmark logs from %s", len(darshan_files), log_dir)

    records = []
    stats = {'total': 0, 'construction': 0, 'relabeled': 0, 'inferred': 0,
             'verified': 0, 'failed_parse': 0}

    for log_path in darshan_files:
        stats['total'] += 1

        # Parse and extract features
        try:
            from src.data.parse_darshan import parse_darshan_log
            from src.data.feature_extraction import extract_raw_features

            report = parse_darshan_log(str(log_path))
            features = extract_raw_features(report)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", log_path.name, e)
            stats['failed_parse'] += 1
            continue

        # Assign ground-truth label
        labels, source, verified = assign_groundtruth_label(
            log_path.name, darshan_features=features
        )
        stats[source] = stats.get(source, 0) + 1
        if verified:
            stats['verified'] += 1

        # Build record
        record = {
            'filename': log_path.name,
            'path': str(log_path),
            'benchmark': _extract_benchmark_type(log_path.name),
            'scenario': _extract_scenario(log_path.name),
            'label_source': source,
            'verified': verified,
            'nprocs': features.get('nprocs', 0),
            'runtime_seconds': features.get('runtime_seconds', 0),
        }
        for dim in DIMENSION_NAMES:
            record[dim] = labels[dim]

        records.append(record)

    if not records:
        logger.error("No records produced")
        return None

    df = pd.DataFrame(records)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Ground-truth labels written to %s", output_path)
    logger.info("Statistics: %s", stats)
    logger.info("Label distribution:")
    for dim in DIMENSION_NAMES:
        count = df[dim].sum()
        logger.info("  %s: %d (%.1f%%)", dim, count, 100 * count / len(df))

    return df


def _extract_benchmark_type(filename):
    """Extract benchmark type from filename."""
    name = filename.lower()
    # Order matters: check specific patterns before generic ones
    if 'hacc_io' in name:
        return 'hacc_io'
    elif 'h5bench' in name or 'h5b_' in name:
        return 'h5bench'
    elif 'mdtest' in name:
        return 'mdtest'
    elif name.startswith('ior_') or '_ior_' in name or '_ior ' in name:
        return 'ior'
    elif 'dlio' in name:
        return 'dlio'
    elif 'custom_' in name or 'imbalance' in name or 'balanced' in name:
        return 'custom'
    return 'unknown'


def _extract_scenario(filename):
    """Extract scenario name from filename."""
    name = os.path.basename(filename).lower()
    # Remove extension and job ID suffix
    name = re.sub(r'_\d{4,}_\d+.*\.darshan$', '', name)
    name = re.sub(r'\.darshan$', '', name)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth labels from benchmark Darshan logs"
    )
    parser.add_argument(
        '--log-dir', type=str, required=True,
        help='Root directory containing benchmark .darshan files'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for labels file (.csv or .parquet)'
    )
    parser.add_argument(
        '--format', type=str, default='csv', choices=['csv', 'parquet'],
        help='Output format (default: csv)'
    )
    args = parser.parse_args()

    process_benchmark_logs(args.log_dir, args.output, args.format)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    main()
