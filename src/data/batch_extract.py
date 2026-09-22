"""
Batch Feature Extraction
=========================
Parallel Stage 1 extraction of raw features from many .darshan files.

Produces one Parquet file with one row per log: the raw counters, indicators
and metadata of ``extract_raw_features`` plus ``_source_path``.

Designed for scale (1M+ files):
  - multiprocessing.Pool with imap_unordered (lazy, memory-efficient)
  - maxtasksperchild for automatic worker recycling (C library leaks)
  - Per-file signal.alarm timeout (prevents hung PyDarshan C calls)
  - Atomic writes (write to .tmp, rename to final)
  - Resume by identity: a rerun skips every path already present in a part
    file and retries the rest, so no input is processed twice or lost
  - Error CSV with the original exception per failed path
  - Accounting: the final file is written only when every requested path has
    exactly one terminal record (a row or an error), and the part files are
    kept next to it as evidence

Exit status of the CLI: 0 when every file was extracted, 3 when some files
failed but the accounting is complete (read the errors CSV), 1 otherwise.

Usage::

    python -m src.data.batch_extract \\
        --file-list /path/chunk_042.txt \\
        --output /path/chunks/chunk_042.parquet \\
        --workers 100 --timeout 120
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

from src.data.feature_extraction import extract_raw_features
from src.data.parse_darshan import parse_darshan_log

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_PARTIAL = 3


class ExtractionError(RuntimeError):
    """The batch cannot be published (no input, no success, or inconsistent accounting)."""


# ---------------------------------------------------------------------------
# Per-file extraction (runs in worker processes)
# ---------------------------------------------------------------------------

class _FileTimeout(Exception):
    """Raised when a single file exceeds the timeout."""


def _alarm_handler(signum, frame):
    raise _FileTimeout("File processing timed out")


def extract_single_log(darshan_path):
    """Raw features of one .darshan file; raises on any parse failure."""
    parsed = parse_darshan_log(darshan_path, strict=True)
    features = extract_raw_features(parsed)
    features['_source_path'] = str(darshan_path)
    return features


def _extract_with_timeout(args):
    """Run ``extract_single_log`` under a per-file alarm in a worker process.

    Returns ``(features_or_None, error_or_None, path)``; the error is the
    original exception text.
    """
    darshan_path, timeout_sec = args
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(timeout_sec)
        result = extract_single_log(darshan_path)
        signal.alarm(0)
        return result, None, darshan_path
    except _FileTimeout:
        return None, f"timeout_after_{timeout_sec}s", darshan_path
    except Exception as exc:
        signal.alarm(0)
        return None, f"{type(exc).__name__}: {str(exc)[:300]}", darshan_path
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Part files
# ---------------------------------------------------------------------------

def _part_path(chunk_dir, stem, suffix, part_idx):
    """Sub-chunk path; the ``_part_`` infix keeps it apart from the SLURM-level
    ``chunk_NNN.parquet`` files."""
    return chunk_dir / f"{stem}_part_{part_idx:04d}{suffix}"


def _existing_parts(chunk_dir, stem, suffix):
    return sorted(chunk_dir.glob(f"{stem}_part_[0-9][0-9][0-9][0-9]{suffix}"))


def _write_parquet(df, path):
    """Atomic write: to a temporary file, then rename."""
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    df.to_parquet(tmp_path, index=False, engine='pyarrow')
    os.rename(tmp_path, path)


def _write_part(records, part_path):
    df = pd.DataFrame(records)
    _write_parquet(df, part_path)
    logger.info("Wrote part: %s (%d rows, %.1f MB)",
                part_path.name, len(df), part_path.stat().st_size / 1e6)


def _publish(parts, requested, failed, output_path):
    """Concatenate the parts into the final file after checking the accounting.

    Every requested path must appear exactly once, either as a row or as a
    failure. The parts stay in place as the evidence for the final file.
    """
    frames = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise ExtractionError("no successful extraction; nothing to publish")

    rows = merged['_source_path']
    duplicates = rows[rows.duplicated()].unique()
    if len(duplicates):
        raise ExtractionError(f"{len(duplicates)} paths appear in more than one part, "
                              f"first: {duplicates[0]}")
    done = set(rows)
    both = done & failed
    if both:
        raise ExtractionError(f"{len(both)} paths are both extracted and failed, "
                              f"first: {sorted(both)[0]}")
    unknown = (done | failed) - requested
    if unknown:
        raise ExtractionError(f"{len(unknown)} extracted paths were not requested, "
                              f"first: {sorted(unknown)[0]}")
    missing = requested - done - failed
    if missing:
        raise ExtractionError(f"{len(missing)} requested paths have no terminal record, "
                              f"first: {sorted(missing)[0]}")

    _write_parquet(merged, output_path)
    logger.info("Published %d rows to %s (%.1f MB) from %d parts",
                len(merged), output_path, output_path.stat().st_size / 1e6, len(parts))
    return len(merged)


# ---------------------------------------------------------------------------
# Main batch extraction
# ---------------------------------------------------------------------------

def batch_extract(input_dir=None, file_list=None, output_path=None,
                  max_workers=60, max_files=None, chunk_size=10000,
                  timeout_per_file=120, shuffle=True, resume=True):
    """Extract raw features from .darshan files in parallel.

    Parameters
    ----------
    input_dir : str or Path, optional
        Directory containing .darshan files (searched recursively).
    file_list : str or Path, optional
        Text file with one .darshan path per line.
    output_path : str or Path
        Output Parquet file path. Part files and ``<stem>_errors.csv`` are
        written next to it.
    max_workers : int
        Number of parallel workers.
    max_files : int, optional
        Maximum files to process (for testing).
    chunk_size : int
        Write a part every N successful extractions.
    timeout_per_file : int
        Seconds before a stuck file is abandoned.
    shuffle : bool
        Shuffle the file list for Lustre MDT load balancing.
    resume : bool
        Skip the paths already present in part files; retry everything else.

    Returns
    -------
    dict
        ``n_requested``, ``n_success``, ``n_failed``, ``n_skipped`` (already
        in parts), ``n_rows`` published.

    Raises
    ------
    ExtractionError
        No input, no successful extraction, or inconsistent accounting; the
        final file is not written in these cases.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix, chunk_dir = output_path.stem, output_path.suffix, output_path.parent

    # --- Collect file list ---
    if file_list is not None:
        with open(file_list) as fh:
            files = [line.strip() for line in fh if line.strip()]
        logger.info("Loaded %d paths from %s", len(files), file_list)
    elif input_dir is not None:
        files = [str(f) for f in sorted(Path(input_dir).rglob('*.darshan'))]
        logger.info("Found %d .darshan files under %s", len(files), input_dir)
    else:
        raise ExtractionError("one of input_dir or file_list is required")
    if max_files is not None:
        files = files[:max_files]
    requested = set(files)
    if len(requested) != len(files):
        raise ExtractionError(f"{len(files) - len(requested)} duplicate paths in the input list")
    if not files:
        raise ExtractionError("no .darshan files to process")

    # --- Resume by identity ---
    parts = _existing_parts(chunk_dir, stem, suffix)
    if parts and not resume:
        raise ExtractionError(f"{len(parts)} part files already in {chunk_dir}; "
                              "resume, or use another output directory")
    already = set()
    for part in parts:
        already.update(pd.read_parquet(part, columns=['_source_path'])['_source_path'])
    stale = already - requested
    if stale:
        raise ExtractionError(f"{len(stale)} paths in existing parts are not in the input "
                              f"(different run in this directory?), first: {sorted(stale)[0]}")
    todo = [f for f in files if f not in already]
    if already:
        logger.info("Resuming: %d paths already extracted in %d parts, %d to do",
                    len(already), len(parts), len(todo))
    part_idx = (max(int(p.stem.rsplit('_', 1)[1]) for p in parts) + 1) if parts else 0

    if shuffle:
        random.seed(42)
        random.shuffle(todo)

    # --- Error log: one row per failed path of this run ---
    error_path = chunk_dir / f"{stem}_errors.csv"
    failed = set()
    n_success = 0
    t_start = time.time()
    logger.info("Starting extraction: %d files, %d workers, chunk_size=%d, timeout=%ds",
                len(todo), max_workers, chunk_size, timeout_per_file)

    with open(error_path, 'a', newline='') as error_file:
        error_writer = csv.writer(error_file)
        if error_path.stat().st_size == 0:
            error_writer.writerow(['file_path', 'error', 'timestamp'])

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(todo), unit='file', desc='Extracting',
                        dynamic_ncols=True, miniters=1)
        except ImportError:
            pbar = None

        results = []
        if todo:
            task_args = [(f, timeout_per_file) for f in todo]
            imap_chunksize = max(1, min(50, len(todo) // (max_workers * 4)))
            # maxtasksperchild recycles workers so C library memory does not grow
            with multiprocessing.Pool(processes=max_workers, maxtasksperchild=500) as pool:
                for result, error, fpath in pool.imap_unordered(
                        _extract_with_timeout, task_args, chunksize=imap_chunksize):
                    if result is not None:
                        results.append(result)
                        n_success += 1
                    else:
                        failed.add(fpath)
                        error_writer.writerow([fpath, error, time.strftime('%Y-%m-%d %H:%M:%S')])
                    if pbar is not None:
                        rate = (n_success + len(failed)) / max(time.time() - t_start, 0.001)
                        pbar.set_postfix_str(f"ok={n_success} fail={len(failed)} rate={rate:.0f}/s",
                                             refresh=False)
                        pbar.update(1)
                    if len(results) >= chunk_size:
                        _write_part(results, _part_path(chunk_dir, stem, suffix, part_idx))
                        part_idx += 1
                        results = []
        if results:
            _write_part(results, _part_path(chunk_dir, stem, suffix, part_idx))
            part_idx += 1
        if pbar is not None:
            pbar.close()

    elapsed = time.time() - t_start
    logger.info("Extraction run: %d success, %d failed, %.1f s (%.1f files/s)",
                n_success, len(failed), elapsed, (n_success + len(failed)) / max(elapsed, 0.001))
    if failed:
        logger.warning("%d files failed; causes in %s", len(failed), error_path)

    n_rows = _publish(_existing_parts(chunk_dir, stem, suffix), requested, failed, output_path)
    return {'n_requested': len(files), 'n_success': n_success, 'n_failed': len(failed),
            'n_skipped': len(already), 'n_rows': n_rows}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Batch extract raw features from Darshan logs')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', help='Directory containing .darshan files')
    input_group.add_argument('--file-list', help='Text file with one .darshan path per line')
    parser.add_argument('--output', required=True, help='Output Parquet file path')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max-files', type=int, default=None, help='For testing')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Write a part every N successful extractions')
    parser.add_argument('--timeout', type=int, default=120, help='Per-file timeout in seconds')
    parser.add_argument('--no-shuffle', action='store_true', help='Keep the input order')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore existing parts (they must not be in the output directory)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    logging.getLogger('darshan').setLevel(logging.WARNING)

    try:
        summary = batch_extract(
            input_dir=args.input_dir, file_list=args.file_list, output_path=args.output,
            max_workers=args.workers, max_files=args.max_files, chunk_size=args.chunk_size,
            timeout_per_file=args.timeout, shuffle=not args.no_shuffle,
            resume=not args.no_resume)
    except ExtractionError as exc:
        logger.error("EXTRACTION FAILED: %s", exc)
        return EXIT_FATAL
    logger.info("Summary: %s", summary)
    return EXIT_PARTIAL if summary['n_failed'] else EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
