"""
Batch Feature Extraction
=========================
Parallel extraction of features from a directory of .darshan files.

Produces a single Parquet file with one row per job, containing:
  - ~136 ML features (raw counters + derived ratios)
  - Job metadata columns (_jobid, _uid, _start_time, _modules, _uses_lustre)

Usage::

    python -m src.data.batch_extract \\
        --input-dir Darshan_Logs/2024/ \\
        --output data/processed/production_features_2024.parquet \\
        --workers 16 \\
        --max-files 50000
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data.parse_darshan import parse_darshan_log
from src.data.aggregate import aggregate_from_total_output
from src.data.feature_extraction import extract_features

logger = logging.getLogger(__name__)


def extract_single_log(darshan_path, backend=None):
    """Extract features from one .darshan file.

    Parameters
    ----------
    darshan_path : str
        Path to .darshan file.
    backend : str, optional
        Parser backend ('pydarshan' or 'cli').

    Returns
    -------
    dict or None
        Feature dictionary, or None if parsing fails.
    """
    try:
        parsed = parse_darshan_log(darshan_path, backend=backend)
        if parsed is None:
            return None

        # Aggregation (--total output is already aggregated)
        parsed['counters'] = aggregate_from_total_output(parsed['counters'])

        # Feature extraction
        features = extract_features(parsed, apply_log_transform=True)
        features['_source_path'] = str(darshan_path)
        return features

    except Exception:
        logger.debug("Failed: %s", darshan_path, exc_info=True)
        return None


def batch_extract(input_dir, output_path, max_workers=16, max_files=None,
                  backend=None, chunk_size=10000):
    """Extract features from all .darshan files in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .darshan files (searched recursively).
    output_path : str or Path
        Output Parquet file path.
    max_workers : int
        Number of parallel workers.
    max_files : int, optional
        Maximum files to process (for testing).
    backend : str, optional
        Parser backend.
    chunk_size : int
        Write intermediate results every chunk_size files.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all .darshan files
    files = sorted(input_dir.rglob('*.darshan'))
    if max_files is not None:
        files = files[:max_files]
    n_total = len(files)
    logger.info("Found %d .darshan files in %s", n_total, input_dir)

    if n_total == 0:
        logger.warning("No .darshan files found!")
        return

    results = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {
            pool.submit(extract_single_log, str(f), backend): f
            for f in files
        }

        for i, future in enumerate(as_completed(future_to_path)):
            r = future.result()
            if r is not None:
                results.append(r)
                n_success += 1
            else:
                n_fail += 1

            # Progress logging
            if (i + 1) % 1000 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) | success=%d fail=%d | "
                    "%.1f logs/s | ETA: %.0fs",
                    i + 1, n_total, 100 * (i + 1) / n_total,
                    n_success, n_fail, rate, eta
                )

            # Write intermediate chunks to avoid memory issues
            if len(results) >= chunk_size:
                _append_to_parquet(results, output_path)
                results = []

    # Write remaining results
    if results:
        _append_to_parquet(results, output_path)

    elapsed = time.time() - t_start
    logger.info(
        "Batch extraction complete: %d success, %d failed, %.1f seconds (%.1f logs/s)",
        n_success, n_fail, elapsed, n_success / max(elapsed, 0.001)
    )
    logger.info("Output: %s", output_path)


def _append_to_parquet(records, output_path):
    """Append records to a Parquet file (create if not exists)."""
    df = pd.DataFrame(records)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(output_path, index=False)
    logger.debug("Written %d total rows to %s", len(df), output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Batch extract features from Darshan logs'
    )
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing .darshan files')
    parser.add_argument('--output', required=True,
                        help='Output Parquet file path')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to process')
    parser.add_argument('--backend', choices=['pydarshan', 'cli'], default=None,
                        help='Parser backend (auto-detect if not specified)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Write intermediate results every N files')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    batch_extract(
        input_dir=args.input_dir,
        output_path=args.output,
        max_workers=args.workers,
        max_files=args.max_files,
        backend=args.backend,
        chunk_size=args.chunk_size,
    )


if __name__ == '__main__':
    main()
