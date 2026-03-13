#!/usr/bin/env python3
"""Generate load-imbalanced I/O for parallelism_efficiency bottleneck ground truth.

Rank 0 writes `imbalance_factor` times more data than other ranks.
This creates a clear imbalance in POSIX_BYTES_WRITTEN across ranks,
which Darshan captures as high rank_bytes_cv and rank_time_cv.

Usage (via srun with Darshan LD_PRELOAD):
    srun --export=ALL,LD_PRELOAD=/path/to/libdarshan.so \
        python benchmarks/custom/load_imbalance.py \
        --imbalance-factor 10 --base-size-mb 10 --output-dir /path/to/output

Ground-truth label: parallelism_efficiency = 1 (by construction)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate load-imbalanced parallel I/O workload"
    )
    parser.add_argument(
        "--imbalance-factor", type=float, default=10.0,
        help="Rank 0 writes this many times more data (default: 10)"
    )
    parser.add_argument(
        "--base-size-mb", type=int, default=10,
        help="Base data size per non-zero rank in MB (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    try:
        from mpi4py import MPI
    except ImportError:
        logger.error("mpi4py not installed. Build from source: MPICC=cc pip install --no-binary mpi4py mpi4py")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        logger.info(
            "Load imbalance benchmark: %d ranks, factor=%.1f, base=%d MB",
            size, args.imbalance_factor, args.base_size_mb
        )

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed + rank)

    # Rank 0 writes imbalance_factor * base_size, others write base_size
    if rank == 0:
        data_size = int(args.base_size_mb * args.imbalance_factor * 1024 * 1024)
    else:
        data_size = int(args.base_size_mb * 1024 * 1024)

    # Write in chunks to avoid memory issues with large data
    chunk_size = min(data_size, 64 * 1024 * 1024)  # 64MB max chunk
    remaining = data_size
    output_file = os.path.join(args.output_dir, f"imbalance_rank{rank:04d}.dat")

    t_start = time.time()
    with open(output_file, 'wb') as f:
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            data = np.random.bytes(write_size)
            f.write(data)
            remaining -= write_size
            f.flush()
    # fsync to ensure data reaches disk
    with open(output_file, 'rb') as f:
        os.fsync(f.fileno())
    t_write = time.time() - t_start

    # Read back the file (for read imbalance too)
    t_start = time.time()
    with open(output_file, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
    t_read = time.time() - t_start

    comm.Barrier()

    if rank == 0:
        logger.info(
            "Rank 0: wrote %.1f MB in %.2fs, read in %.2fs",
            data_size / 1e6, t_write, t_read
        )
    else:
        logger.info(
            "Rank %d: wrote %.1f MB in %.2fs, read in %.2fs",
            rank, data_size / 1e6, t_write, t_read
        )

    comm.Barrier()

    # Cleanup
    os.remove(output_file)

    if rank == 0:
        logger.info("Benchmark complete. Imbalance factor: %.1f", args.imbalance_factor)

    MPI.Finalize()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    main()
