"""
Batch Feature Extraction
=========================
Parallel extraction of features from a directory of .darshan files.

Produces Parquet file(s) with one row per job, containing:
  - ~147 ML features (raw counters + indicators + metadata)
  - Job info columns (_jobid, _uid, _start_time, _modules, etc.)

Designed for scale (1M+ files):
  - multiprocessing.Pool with imap_unordered (lazy, memory-efficient)
  - maxtasksperchild for automatic worker recycling (C library leaks)
  - Per-file signal.alarm timeout (prevents hung PyDarshan C calls)
  - Atomic writes (write to .tmp, rename to final)
  - Checkpoint/resume (skip existing sub-chunks)
  - Error CSV logging (failed paths + error messages)
  - tqdm progress bar with live stats

Usage::

    # Full directory scan
    python -m src.data.batch_extract \\
        --input-dir Darshan_Logs/2024/ \\
        --output data/processed/raw_features_2024.parquet \\
        --workers 16 --raw

    # From file list (for SLURM array jobs)
    python -m src.data.batch_extract \\
        --file-list /tmp/chunk_042.txt \\
        --output data/processed/chunks/chunk_042.parquet \\
        --workers 100 --raw --timeout 120
"""

import argparse
import csv
import logging
import multiprocessing
import os
import random
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features, extract_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-file extraction (runs in worker processes)
# ---------------------------------------------------------------------------


class _FileTimeout(Exception):
    """Raised when a single file exceeds the timeout."""
    pass


def _alarm_handler(signum, frame):
    """Signal handler for SIGALRM — raises _FileTimeout."""
    raise _FileTimeout("File processing timed out")


def extract_single_log(darshan_path, backend=None, raw_only=False):
    """Extract features from one .darshan file.

    Parameters
    ----------
    darshan_path : str
        Path to .darshan file.
    backend : str, optional
        Parser backend ('pydarshan' or 'cli').
    raw_only : bool
        If True, extract raw features only (no transforms, no derived).

    Returns
    -------
    dict or None
        Feature dictionary, or None if parsing fails.
    """
    try:
        parsed = parse_darshan_log(darshan_path, backend=backend)
        if parsed is None:
            return None

        if raw_only:
            features = extract_raw_features(parsed)
        else:
            features = extract_features(parsed, apply_log_transform=True)

        features['_source_path'] = str(darshan_path)
        return features

    except Exception:
        logger.debug("Failed: %s", darshan_path, exc_info=True)
        return None


def _extract_with_timeout(args):
    """Wrapper that adds per-file timeout via signal.alarm.

    This runs in a forked worker process, so each gets its own signal handler.
    Returns (result_dict_or_None, error_string_or_None, file_path).
    """
    darshan_path, backend, raw_only, timeout_sec = args

    # Install alarm handler (safe in forked child — each has own signals)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(timeout_sec)
        result = extract_single_log(darshan_path, backend=backend,
                                    raw_only=raw_only)
        signal.alarm(0)  # Cancel alarm on success

        if result is None:
            return None, "parse_returned_none", darshan_path
        return result, None, darshan_path

    except _FileTimeout:
        return None, f"timeout_after_{timeout_sec}s", darshan_path

    except Exception as exc:
        signal.alarm(0)
        return None, str(exc)[:200], darshan_path

    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Chunk I/O helpers
# ---------------------------------------------------------------------------

def _write_chunk(records, chunk_path):
    """Write a list of record dicts as a Parquet file (atomic)."""
    tmp_path = chunk_path.with_suffix('.parquet.tmp')
    df = pd.DataFrame(records)
    df.to_parquet(tmp_path, index=False, engine='pyarrow')
    os.rename(tmp_path, chunk_path)
    logger.info("Wrote chunk: %s (%d rows, %.1f MB)",
                chunk_path.name, len(df),
                chunk_path.stat().st_size / 1e6)


def _get_part_path(chunk_dir, stem, suffix, part_idx):
    """Get path for internal sub-chunk: uses _part_ prefix to avoid
    collision with SLURM-level chunk_NNN.parquet files."""
    return chunk_dir / f"{stem}_part_{part_idx:04d}{suffix}"


def _merge_internal_parts(chunk_dir, stem, suffix, n_parts, output_path):
    """Merge all internal _part_ files into single output."""
    logger.info("Merging %d internal parts into %s...", n_parts, output_path)
    t_start = time.time()

    dfs = []
    for i in range(n_parts):
        part_path = _get_part_path(chunk_dir, stem, suffix, i)
        if part_path.exists():
            dfs.append(pd.read_parquet(part_path))
        else:
            logger.warning("Missing part file: %s", part_path.name)

    if not dfs:
        logger.warning("No part files found to merge!")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Atomic write for final output
    tmp_path = output_path.with_suffix('.parquet.tmp')
    merged.to_parquet(tmp_path, index=False, engine='pyarrow')
    os.rename(tmp_path, output_path)

    elapsed = time.time() - t_start
    logger.info("Merged %d rows into %s (%.1fs, %.1f MB)",
                len(merged), output_path.name, elapsed,
                output_path.stat().st_size / 1e6)

    # Clean up part files
    cleaned = 0
    for i in range(n_parts):
        part_path = _get_part_path(chunk_dir, stem, suffix, i)
        if part_path.exists():
            part_path.unlink()
            cleaned += 1
    logger.info("Cleaned up %d part files", cleaned)


# ---------------------------------------------------------------------------
# Main batch extraction
# ---------------------------------------------------------------------------

def batch_extract(input_dir=None, file_list=None, output_path=None,
                  max_workers=60, max_files=None, backend=None,
                  chunk_size=10000, raw_only=False,
                  timeout_per_file=120, shuffle=True, resume=True):
    """Extract features from .darshan files in parallel.

    Uses multiprocessing.Pool with imap_unordered for lazy task submission
    and maxtasksperchild for automatic worker recycling.

    Parameters
    ----------
    input_dir : str or Path, optional
        Directory containing .darshan files (searched recursively).
    file_list : str or Path, optional
        Text file with one .darshan path per line.
    output_path : str or Path
        Output Parquet file path.
    max_workers : int
        Number of parallel workers.
    max_files : int, optional
        Maximum files to process (for testing).
    backend : str, optional
        Parser backend.
    chunk_size : int
        Write a sub-chunk every N successful extractions.
    raw_only : bool
        If True, extract raw features only (Stage 1).
    timeout_per_file : int
        Seconds before killing a stuck worker via signal.alarm.
    shuffle : bool
        Shuffle file list for Lustre MDT load balancing.
    resume : bool
        Skip already-completed sub-chunks on restart.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode_label = "RAW" if raw_only else "FULL"
    logger.info("Extraction mode: %s", mode_label)

    # --- Collect file list ---
    if file_list is not None:
        file_list_path = Path(file_list)
        with open(file_list_path) as fh:
            files = [line.strip() for line in fh if line.strip()]
        logger.info("Loaded %d paths from %s", len(files), file_list_path)
    elif input_dir is not None:
        input_dir = Path(input_dir)
        logger.info("Scanning %s for .darshan files...", input_dir)
        files = [str(f) for f in sorted(input_dir.rglob('*.darshan'))]
        logger.info("Found %d .darshan files", len(files))
    else:
        logger.error("Must specify --input-dir or --file-list")
        return

    if max_files is not None:
        files = files[:max_files]
    n_total = len(files)

    if n_total == 0:
        logger.warning("No .darshan files found!")
        return

    # Shuffle for Lustre MDT load balancing
    if shuffle:
        random.seed(42)
        random.shuffle(files)
        logger.info("Shuffled file list (seed=42) for MDT load balancing")

    # --- Checkpoint / resume support ---
    stem = output_path.stem
    suffix = output_path.suffix
    chunk_dir = output_path.parent

    # Figure out how many complete sub-chunks already exist
    n_parts_done = 0
    if resume:
        while True:
            part_path = _get_part_path(chunk_dir, stem, suffix, n_parts_done)
            if part_path.exists():
                n_parts_done += 1
            else:
                break
        if n_parts_done > 0:
            n_skip = n_parts_done * chunk_size
            if n_skip >= n_total:
                logger.info("All %d files already processed (%d parts). "
                            "Skipping to merge.", n_total, n_parts_done)
                _merge_internal_parts(chunk_dir, stem, suffix, n_parts_done,
                                      output_path)
                return
            files = files[n_skip:]
            logger.info("Resuming: skipping %d files (%d complete parts), "
                        "%d files remaining", n_skip, n_parts_done, len(files))

    n_remaining = len(files)
    part_idx = n_parts_done

    # --- Error logging setup ---
    error_path = chunk_dir / f"{stem}_errors.csv"
    error_file = open(error_path, 'a', newline='')
    error_writer = csv.writer(error_file)
    if error_path.stat().st_size == 0:
        error_writer.writerow(['file_path', 'error', 'timestamp'])

    # --- Progress bar (tqdm) ---
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_remaining, unit='file', desc='Extracting',
                    dynamic_ncols=True, miniters=1)
    except ImportError:
        pbar = None
        logger.warning("tqdm not available, using log-only progress")

    # --- Process with multiprocessing.Pool ---
    results = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    logger.info("Starting extraction: %d files, %d workers, chunk_size=%d, "
                "timeout=%ds, mode=%s",
                n_remaining, max_workers, chunk_size,
                timeout_per_file, mode_label)

    # Build argument tuples for imap_unordered
    task_args = [(f, backend, raw_only, timeout_per_file) for f in files]

    # maxtasksperchild=500 recycles workers to prevent C library memory leaks
    with multiprocessing.Pool(processes=max_workers,
                              maxtasksperchild=500) as pool:
        # chunksize=50 amortizes IPC overhead without buffering too much
        imap_chunksize = max(1, min(50, n_remaining // (max_workers * 4)))

        for result, error, fpath in pool.imap_unordered(
                _extract_with_timeout, task_args, chunksize=imap_chunksize):

            if result is not None:
                results.append(result)
                n_success += 1
            else:
                n_fail += 1
                error_writer.writerow([fpath, error or 'unknown',
                                       time.strftime('%Y-%m-%d %H:%M:%S')])

            # Update progress bar
            if pbar is not None:
                elapsed = time.time() - t_start
                rate = (n_success + n_fail) / max(elapsed, 0.001)
                pbar.set_postfix_str(
                    f"ok={n_success} fail={n_fail} "
                    f"rate={rate:.0f}/s",
                    refresh=False)
                pbar.update(1)

            # Write sub-chunk when buffer is full
            if len(results) >= chunk_size:
                part_path = _get_part_path(chunk_dir, stem, suffix, part_idx)
                _write_chunk(results, part_path)
                part_idx += 1
                results = []

    # Write remaining results
    if results:
        part_path = _get_part_path(chunk_dir, stem, suffix, part_idx)
        _write_chunk(results, part_path)
        part_idx += 1

    # Close progress bar and error file
    if pbar is not None:
        pbar.close()
    error_file.close()

    n_parts_total = part_idx
    elapsed = time.time() - t_start
    total_processed = n_success + n_fail

    logger.info(
        "Extraction complete (%s): %d success, %d failed (%.1f%% success), "
        "%d parts written, %.1f seconds (%.1f files/s)",
        mode_label, n_success, n_fail,
        100 * n_success / max(total_processed, 1),
        n_parts_total, elapsed,
        total_processed / max(elapsed, 0.001)
    )

    # Report error summary
    if n_fail > 0:
        logger.info("Error details written to: %s", error_path)

    # Merge all parts into final output
    if n_parts_total > 0:
        _merge_internal_parts(chunk_dir, stem, suffix, n_parts_total,
                              output_path)
    else:
        logger.warning("No parts written — all files failed!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch extract features from Darshan logs'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir',
                             help='Directory containing .darshan files')
    input_group.add_argument('--file-list',
                             help='Text file with one .darshan path per line')
    parser.add_argument('--output', required=True,
                        help='Output Parquet file path')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to process (for testing)')
    parser.add_argument('--backend', choices=['pydarshan', 'cli'],
                        default=None,
                        help='Parser backend (auto-detect if not specified)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Write a sub-chunk every N extractions')
    parser.add_argument('--raw', action='store_true',
                        help='Extract raw features only (Stage 1)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-file timeout in seconds (default: 120)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable file list shuffling')
    parser.add_argument('--no-resume', action='store_true',
                        help='Disable checkpoint/resume (reprocess all)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )
    # Suppress noisy libraries
    logging.getLogger('darshan').setLevel(logging.WARNING)

    batch_extract(
        input_dir=args.input_dir,
        file_list=args.file_list,
        output_path=args.output,
        max_workers=args.workers,
        max_files=args.max_files,
        backend=args.backend,
        chunk_size=args.chunk_size,
        raw_only=args.raw,
        timeout_per_file=args.timeout,
        shuffle=not args.no_shuffle,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()
