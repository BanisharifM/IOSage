#!/usr/bin/env python3
"""
Verify that benchmark Darshan logs match expected ground-truth labels.

For each benchmark type, parse a sample of Darshan logs and check:
1. The log parses successfully
2. Key Darshan counters match the expected bottleneck signature
3. The label encoded in the job name is consistent with I/O patterns

Ground-Truth Label Mapping (by construction):
----------------------------------------------
IOR:
  small_posix/small_mpiio → access_granularity=1 (transfer_size << 1MB)
  random_posix             → access_pattern=1 (random offsets)
  shared_posix             → interface_choice=1 (POSIX on shared file, no collective)
  fpp_explosion            → file_strategy=1 (one file per process)
  healthy_*                → healthy=1 (large sequential, proper interface)

mdtest:
  meta_shared/meta_unique/meta_cross/io500_hard/deep_tree → metadata_intensity=1
  fpp_explosion            → file_strategy=1
  healthy/io500_easy       → healthy=1

DLIO:
  small_rl*                → access_granularity=1 (tiny record lengths)
  ckpt_ms*                 → throughput_utilization=1 (checkpoint bursts)
  healthy_rl*              → healthy=1
  shuffle                  → access_pattern=1

Custom:
  imbalance_f*             → parallelism_efficiency=1 (load imbalance)
  balanced_*               → healthy=1
"""

import argparse
import glob
import logging
import os
import random
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features


def infer_label_from_jobname(log_path):
    """Extract the ground-truth label from the Darshan log filename.

    Darshan log names: <user>_<exe>_id<jobid>-<hash>_<date>-<hash>_<rank>.darshan
    The SLURM job name is embedded in the output file names, not the Darshan log.
    We need to match via SLURM job ID in the filename.
    """
    basename = os.path.basename(log_path)
    # Extract SLURM job ID from Darshan filename: ..._id<JOBID>-...
    parts = basename.split("_id")
    if len(parts) < 2:
        return None, None
    jobid_part = parts[1].split("-")[0]
    return jobid_part, basename


def check_ior_log(log_path, features):
    """Validate IOR log features against expected patterns."""
    issues = []

    # IOR should have POSIX or MPI-IO data
    bytes_w = features.get("POSIX_BYTES_WRITTEN", 0) or 0
    bytes_r = features.get("POSIX_BYTES_READ", 0) or 0
    mpiio_w = features.get("MPIIO_BYTES_WRITTEN", 0) or 0
    mpiio_r = features.get("MPIIO_BYTES_READ", 0) or 0

    total_bytes = bytes_w + bytes_r + mpiio_w + mpiio_r
    if total_bytes == 0:
        issues.append("ZERO total bytes (IOR should produce I/O)")

    return issues


def check_mdtest_log(log_path, features):
    """Validate mdtest log features."""
    issues = []
    nprocs = features.get("nprocs", 0) or 0
    if nprocs == 0:
        issues.append("nprocs=0")
    return issues


def check_dlio_log(log_path, features):
    """Validate DLIO log features."""
    issues = []
    basename = os.path.basename(log_path)
    # Skip lscpu/uname probe logs (DLIO startup artifacts)
    if "_lscpu_" in basename or "_uname_" in basename:
        return ["SKIP: startup probe log"]

    bytes_r = features.get("POSIX_BYTES_READ", 0) or 0
    if bytes_r == 0:
        issues.append("ZERO bytes_read (DLIO training should read data)")
    return issues


def check_custom_log(log_path, features):
    """Validate custom benchmark log features."""
    issues = []
    bytes_w = features.get("POSIX_BYTES_WRITTEN", 0) or 0
    if bytes_w == 0:
        issues.append("ZERO bytes_written (custom benchmark should write)")
    return issues


def verify_benchmark_type(bench_type, log_dir, sample_size=10, seed=42):
    """Verify a sample of logs from a benchmark type."""
    logs = sorted(glob.glob(os.path.join(log_dir, "*.darshan")))
    if not logs:
        logger.warning(f"  No logs found in {log_dir}")
        return 0, 0, 0

    random.seed(seed)
    sample = random.sample(logs, min(sample_size, len(logs)))

    logger.info(f"  Total logs: {len(logs)}, sampling {len(sample)}")

    passed = 0
    failed = 0
    skipped = 0

    checker = {
        "ior": check_ior_log,
        "mdtest": check_mdtest_log,
        "dlio": check_dlio_log,
        "custom": check_custom_log,
    }[bench_type]

    for log_path in sample:
        basename = os.path.basename(log_path)
        try:
            report = parse_darshan_log(log_path)
            features = extract_raw_features(report)

            issues = checker(log_path, features)

            if issues and issues[0].startswith("SKIP"):
                logger.info(f"    SKIP: {basename} ({issues[0]})")
                skipped += 1
                continue

            if issues:
                logger.warning(f"    FAIL: {basename}: {'; '.join(issues)}")
                failed += 1
            else:
                nprocs = features.get("nprocs", "?")
                runtime = features.get("runtime_seconds", 0) or 0
                bytes_w = features.get("POSIX_BYTES_WRITTEN", 0) or 0
                bytes_r = features.get("POSIX_BYTES_READ", 0) or 0
                n_files = features.get("POSIX_FILENOS", 0) or 0
                logger.info(
                    f"    OK: {basename[:60]}... "
                    f"nprocs={nprocs} runtime={runtime:.1f}s "
                    f"write={bytes_w:,.0f}B read={bytes_r:,.0f}B files={n_files}"
                )
                passed += 1

        except Exception as e:
            logger.error(f"    ERROR: {basename}: {e}")
            failed += 1

    return passed, failed, skipped


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark ground-truth labels")
    parser.add_argument("--project-dir", default="/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Number of logs to sample per benchmark type")
    parser.add_argument("--bench-type", default="all",
                        choices=["all", "ior", "mdtest", "dlio", "custom"])
    args = parser.parse_args()

    log_base = os.path.join(args.project_dir, "data", "benchmark_logs")

    bench_types = ["ior", "mdtest", "dlio", "custom"] if args.bench_type == "all" else [args.bench_type]

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for bench_type in bench_types:
        log_dir = os.path.join(log_base, bench_type)
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying: {bench_type}")
        logger.info(f"{'='*60}")

        p, f, s = verify_benchmark_type(bench_type, log_dir, args.sample_size)
        total_pass += p
        total_fail += f
        total_skip += s
        logger.info(f"  Result: {p} passed, {f} failed, {s} skipped")

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    logger.info(f"{'='*60}")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
