"""Verify that benchmark Darshan logs exhibit intended I/O patterns.

Every benchmark log must pass verification before ground-truth label assignment.
If verification fails, the log is relabeled based on what Darshan actually shows
(the ML model sees features at inference, so labels must match features).

Usage:
    python -m src.data.benchmark_verify \
        --log-dir data/benchmark_logs/ior \
        --output data/benchmark_logs/verification_report.csv

    python -m src.data.benchmark_verify \
        --log-file data/benchmark_logs/ior/some_log.darshan \
        --expected-label access_granularity=1
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Dimension names (must match drishti_labeling.py and groundtruth_labeling.py)
DIMENSION_NAMES = [
    'access_granularity', 'metadata_intensity', 'parallelism_efficiency',
    'access_pattern', 'interface_choice', 'file_strategy',
    'throughput_utilization', 'healthy',
]

# Verification patterns: what Darshan features SHOULD show for each dimension
# Each check: (feature_name, min_expected, max_expected)
VERIFICATION_CHECKS = {
    'access_granularity': [
        ('small_io_ratio', 0.3, 1.0),
    ],
    'metadata_intensity': [
        ('metadata_time_ratio', 0.05, 1.0),
    ],
    'parallelism_efficiency': [
        ('rank_bytes_cv', 0.2, 100.0),
    ],
    'access_pattern': [
        # Random access means NOT sequential
        ('seq_write_ratio', 0.0, 0.7),
    ],
    'interface_choice': [
        # No collective MPI-IO (or no MPI-IO at all)
        # Checked separately: either MPIIO_COLL_WRITES == 0 or no MPI-IO module
    ],
    'file_strategy': [
        # Many files relative to ranks
    ],
    'throughput_utilization': [
        # Low bandwidth or high write time
    ],
    'healthy': [
        ('small_io_ratio', 0.0, 0.3),
    ],
}

# Minimum thresholds for log inclusion
MIN_RUNTIME_SECONDS = 5.0
MIN_TOTAL_BYTES = 10240  # 10 KB
MIN_IO_OPS = 100


def verify_benchmark_log(features, intended_labels, tolerance=0.2):
    """Check that extracted features match intended benchmark pattern.

    Args:
        features: dict of extracted Darshan features for this log
        intended_labels: dict mapping dimension names to expected values (0 or 1)
        tolerance: fraction of checks that can fail and still pass

    Returns:
        (passed: bool, report: dict with per-check details)
    """
    report = {
        'checks': {},
        'passed_checks': 0,
        'total_checks': 0,
        'inclusion_passed': True,
        'inclusion_reason': '',
    }

    # Step 1: Check minimum inclusion thresholds
    runtime = features.get('runtime_seconds', 0)
    total_bytes = features.get('POSIX_BYTES_READ', 0) + features.get('POSIX_BYTES_WRITTEN', 0)
    total_ops = features.get('POSIX_READS', 0) + features.get('POSIX_WRITES', 0)

    if runtime < MIN_RUNTIME_SECONDS:
        report['inclusion_passed'] = False
        report['inclusion_reason'] = f'runtime={runtime:.1f}s < {MIN_RUNTIME_SECONDS}s'
        return False, report

    if total_bytes < MIN_TOTAL_BYTES:
        report['inclusion_passed'] = False
        report['inclusion_reason'] = f'total_bytes={total_bytes} < {MIN_TOTAL_BYTES}'
        return False, report

    if total_ops < MIN_IO_OPS:
        report['inclusion_passed'] = False
        report['inclusion_reason'] = f'total_ops={total_ops} < {MIN_IO_OPS}'
        return False, report

    # Step 2: Check dimension-specific patterns
    for dim_name, expected_val in intended_labels.items():
        if dim_name not in DIMENSION_NAMES:
            continue
        if expected_val != 1:
            continue  # Only verify dimensions marked as bottleneck

        checks = VERIFICATION_CHECKS.get(dim_name, [])
        for feat_name, lo, hi in checks:
            report['total_checks'] += 1
            val = features.get(feat_name, None)

            if val is None:
                report['checks'][f'{dim_name}/{feat_name}'] = {
                    'status': 'missing',
                    'expected': (lo, hi),
                    'value': None,
                }
            elif lo <= val <= hi:
                report['checks'][f'{dim_name}/{feat_name}'] = {
                    'status': 'pass',
                    'expected': (lo, hi),
                    'value': val,
                }
                report['passed_checks'] += 1
            else:
                report['checks'][f'{dim_name}/{feat_name}'] = {
                    'status': 'fail',
                    'expected': (lo, hi),
                    'value': val,
                }

        # Special checks for specific dimensions
        if dim_name == 'interface_choice':
            report['total_checks'] += 1
            coll_writes = features.get('MPIIO_COLL_WRITES', 0)
            has_mpiio = features.get('has_mpiio', 0)
            if has_mpiio == 0 or coll_writes == 0:
                report['checks']['interface_choice/no_collective'] = {
                    'status': 'pass',
                    'value': f'has_mpiio={has_mpiio}, coll_writes={coll_writes}',
                }
                report['passed_checks'] += 1
            else:
                report['checks']['interface_choice/no_collective'] = {
                    'status': 'fail',
                    'value': f'has_mpiio={has_mpiio}, coll_writes={coll_writes}',
                }

        if dim_name == 'file_strategy':
            report['total_checks'] += 1
            num_files = features.get('num_files', 0)
            nprocs = features.get('nprocs', 1)
            files_per_rank = num_files / max(nprocs, 1)
            if files_per_rank >= 1.0 and num_files >= 50:
                report['checks']['file_strategy/many_files'] = {
                    'status': 'pass',
                    'value': f'num_files={num_files}, nprocs={nprocs}, ratio={files_per_rank:.1f}',
                }
                report['passed_checks'] += 1
            else:
                report['checks']['file_strategy/many_files'] = {
                    'status': 'fail',
                    'value': f'num_files={num_files}, nprocs={nprocs}, ratio={files_per_rank:.1f}',
                }

    # Compute pass/fail
    if report['total_checks'] == 0:
        # No specific checks for this dimension — pass by default
        passed = True
    else:
        pass_rate = report['passed_checks'] / report['total_checks']
        passed = pass_rate >= (1.0 - tolerance)

    if not passed:
        fails = {k: v for k, v in report['checks'].items() if v['status'] != 'pass'}
        logger.warning("Benchmark FAILED verification: %s", fails)

    return passed, report


def verify_log_file(log_path, intended_labels):
    """Parse a single Darshan log and verify it.

    Returns:
        (features, passed, report) or (None, False, error_report) on parse failure
    """
    try:
        # Import here to avoid circular imports
        from src.data.parse_darshan import parse_darshan_log
        from src.data.feature_extraction import extract_raw_features

        report_obj = parse_darshan_log(str(log_path))
        features = extract_raw_features(report_obj)
        passed, report = verify_benchmark_log(features, intended_labels)
        return features, passed, report
    except Exception as e:
        logger.error("Failed to parse %s: %s", log_path, e)
        return None, False, {'error': str(e)}


def parse_label_string(label_str):
    """Parse 'access_granularity=1,access_pattern=1' into dict."""
    labels = {dim: 0 for dim in DIMENSION_NAMES}
    if not label_str:
        return labels
    for pair in label_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, val = pair.split('=', 1)
            key = key.strip()
            if key in DIMENSION_NAMES:
                labels[key] = int(val.strip())
    return labels


def infer_labels_from_filename(filename):
    """Infer intended labels from benchmark output filename.

    Naming convention: {benchmark}_{scenario}_{params}.darshan
    Examples:
        ior_small_posix_t512_n16_r1_*.darshan → access_granularity=1
        mdtest_meta_shared_n5000_*.darshan → metadata_intensity=1
    """
    name = os.path.basename(filename).lower()

    labels = {dim: 0 for dim in DIMENSION_NAMES}

    # IOR scenarios
    if 'small_posix' in name or 'small_direct' in name or 'misaligned' in name:
        labels['access_granularity'] = 1
    elif 'random_posix' in name or 'random_small' in name:
        labels['access_pattern'] = 1
        if 'random_small' in name:
            labels['access_granularity'] = 1
    elif 'interface_posix_shared' in name or 'interface_mpiio_indep' in name:
        labels['interface_choice'] = 1
    elif 'file_explosion' in name:
        labels['file_strategy'] = 1
    elif 'fsync_per_write' in name:
        labels['throughput_utilization'] = 1
    elif 'healthy' in name:
        labels['healthy'] = 1

    # mdtest scenarios
    elif 'meta_shared' in name or 'meta_unique' in name or 'meta_cross' in name or 'deep_tree' in name:
        labels['metadata_intensity'] = 1
    elif 'fpp_explosion' in name:
        labels['file_strategy'] = 1

    # DLIO scenarios
    elif 'dlio_small' in name:
        labels['access_granularity'] = 1
    elif 'dlio_ckpt' in name or 'checkpoint' in name:
        labels['throughput_utilization'] = 1
    elif 'shuffle' in name:
        labels['access_pattern'] = 1

    # Custom scenarios
    elif 'imbalance' in name:
        labels['parallelism_efficiency'] = 1
    elif 'balanced' in name:
        labels['healthy'] = 1

    return labels


def batch_verify(log_dir, output_csv=None, label_str=None):
    """Verify all Darshan logs in a directory.

    Args:
        log_dir: Directory containing .darshan files
        output_csv: Path for verification report CSV
        label_str: Override label for all logs (if None, infer from filename)

    Returns:
        (total, passed, failed, skipped) counts
    """
    log_dir = Path(log_dir)
    darshan_files = sorted(log_dir.glob('**/*.darshan'))

    if not darshan_files:
        logger.warning("No .darshan files found in %s", log_dir)
        return 0, 0, 0, 0

    logger.info("Verifying %d Darshan logs in %s", len(darshan_files), log_dir)

    results = []
    total = passed = failed = skipped = 0

    for log_path in darshan_files:
        total += 1

        # Determine intended labels
        if label_str:
            intended = parse_label_string(label_str)
        else:
            intended = infer_labels_from_filename(log_path.name)

        # Verify
        features, ok, report = verify_log_file(log_path, intended)

        if features is None:
            skipped += 1
            status = 'parse_error'
        elif ok:
            passed += 1
            status = 'pass'
        else:
            failed += 1
            status = 'fail'

        row = {
            'filename': log_path.name,
            'path': str(log_path),
            'status': status,
            'inclusion_passed': report.get('inclusion_passed', False),
            'passed_checks': report.get('passed_checks', 0),
            'total_checks': report.get('total_checks', 0),
        }
        # Add intended labels
        for dim in DIMENSION_NAMES:
            row[f'intended_{dim}'] = intended.get(dim, 0)
        # Add key features
        if features:
            for key in ['nprocs', 'runtime_seconds', 'small_io_ratio',
                        'metadata_time_ratio', 'seq_write_ratio', 'rank_bytes_cv',
                        'write_bw_mb_s', 'num_files']:
                row[key] = features.get(key, '')

        results.append(row)

        log_fn = logger.info if ok else logger.warning
        log_fn(
            "%s: %s (%d/%d checks)",
            log_path.name, status,
            report.get('passed_checks', 0),
            report.get('total_checks', 0)
        )

    # Write CSV report
    if output_csv and results:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Verification report written to %s", output_path)

    logger.info(
        "Verification complete: %d total, %d passed, %d failed, %d skipped",
        total, passed, failed, skipped
    )
    return total, passed, failed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Verify benchmark Darshan logs match intended I/O patterns"
    )
    parser.add_argument(
        '--log-dir', type=str,
        help='Directory containing .darshan files to verify'
    )
    parser.add_argument(
        '--log-file', type=str,
        help='Single .darshan file to verify'
    )
    parser.add_argument(
        '--expected-label', type=str, default=None,
        help='Expected label (e.g., "access_granularity=1,access_pattern=1")'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path for verification report'
    )
    args = parser.parse_args()

    if args.log_file:
        intended = parse_label_string(args.expected_label) if args.expected_label \
            else infer_labels_from_filename(args.log_file)
        features, passed, report = verify_log_file(args.log_file, intended)
        print(f"File: {args.log_file}")
        print(f"Intended: {intended}")
        print(f"Passed: {passed}")
        for check_name, check_info in report.get('checks', {}).items():
            print(f"  {check_name}: {check_info['status']} "
                  f"(value={check_info.get('value')}, "
                  f"expected={check_info.get('expected')})")
    elif args.log_dir:
        batch_verify(args.log_dir, args.output, args.expected_label)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    main()
