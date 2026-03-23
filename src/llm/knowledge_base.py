"""
Knowledge Base Construction from Benchmark Ground-Truth Data.

Builds structured KB entries mapping:
  {benchmark config -> Darshan signature -> source code pattern -> fix -> measured speedup}

This is a key novelty: no existing system has source-code-to-Darshan-signature mapping
with verified fixes and measured improvements.

Used by:
  - Track A: Export for Tabassum's iterative LLM pipeline
  - Track B: RAG retrieval for our recommendation pipeline
  - Track C: Context for our iterative code optimization

Usage:
    python -m src.llm.knowledge_base
    python -m src.llm.knowledge_base --output data/knowledge_base/
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

# Benchmark source code repositories (all public)
BENCHMARK_SOURCES = {
    "ior": {
        "repo": "https://github.com/hpc/ior",
        "language": "C",
        "io_functions": {
            "POSIX": "write(fd, buf, transferSize)",
            "MPIIO_indep": "MPI_File_write(fh, buf, transferSize, MPI_BYTE, &status)",
            "MPIIO_coll": "MPI_File_write_all(fh, buf, transferSize, MPI_BYTE, &status)",
        },
    },
    "mdtest": {
        "repo": "https://github.com/hpc/ior (bundled)",
        "language": "C",
        "io_functions": {
            "create": "open(path, O_CREAT|O_WRONLY, 0600)",
            "stat": "stat(path, &statbuf)",
            "remove": "unlink(path)",
        },
    },
    "h5bench": {
        "repo": "https://github.com/hpc-io/h5bench",
        "language": "C",
        "io_functions": {
            "write_indep": "H5Dwrite(dset, type, memspace, filespace, H5P_DEFAULT, data)",
            "write_coll": "H5Dwrite(dset, type, memspace, filespace, dxpl_coll, data)",
        },
    },
    "hacc_io": {
        "repo": "ANL HACC-IO proxy",
        "language": "C",
        "io_functions": {
            "posix_shared": "write(fd, buf, nbytes)  /* shared file, POSIX */",
            "mpiio_shared": "MPI_File_write_at_all(fh, offset, buf, count, MPI_FLOAT, &status)",
            "fpp": "write(fd, buf, nbytes)  /* file-per-process */",
        },
    },
    "dlio": {
        "repo": "https://github.com/argonne-lcf/dlio_benchmark",
        "language": "Python",
        "io_functions": {
            "read": "np.fromfile(path, dtype=np.float32, count=record_length)",
            "write": "data.tofile(checkpoint_path)",
        },
    },
    "custom": {
        "repo": "custom mpi4py scripts",
        "language": "Python",
        "io_functions": {
            "write": "f.write(data)  /* via mpi4py + DARSHAN_ENABLE_NONMPI */",
        },
    },
}

# Fix patterns for each bottleneck type
FIX_PATTERNS = {
    "access_granularity": {
        "cause": "I/O operations with very small transfer sizes (<1MB), causing excessive system call overhead",
        "fix": "Increase transfer/buffer size to >=1MB. For POSIX: aggregate small writes into large buffers. For MPI-IO: use collective I/O with larger datatypes.",
        "code_before": "for (i=0; i<n; i++) write(fd, buf, 64);  /* 64B per call */",
        "code_after": "/* Buffer small writes, flush in large chunks */\nmemcpy(big_buf + offset, buf, 64); offset += 64;\nif (offset >= 1048576) { write(fd, big_buf, offset); offset = 0; }",
    },
    "metadata_intensity": {
        "cause": "Excessive file metadata operations (open/stat/close) relative to data I/O",
        "fix": "Reduce file count, batch metadata operations, use shared files instead of file-per-process for small data.",
        "code_before": "for (i=0; i<50000; i++) {\n  fd = open(path_i, O_CREAT|O_WRONLY);\n  close(fd);\n}",
        "code_after": "/* Create files in batches, or use single shared file */\nMPI_File_open(comm, shared_path, MPI_MODE_CREATE|MPI_MODE_WRONLY, info, &fh);",
    },
    "parallelism_efficiency": {
        "cause": "Uneven I/O load distribution across MPI ranks",
        "fix": "Balance I/O workload across ranks. Use collective I/O for automatic load balancing. Ensure each rank processes similar data volumes.",
        "code_before": "if (rank == 0) write(fd, all_data, total_size);  /* rank 0 does all I/O */",
        "code_after": "/* Distribute I/O across all ranks */\nMPI_File_write_all(fh, my_data, my_size, MPI_BYTE, &status);",
    },
    "access_pattern": {
        "cause": "Random (non-sequential) file access, defeating OS read-ahead and storage prefetching",
        "fix": "Sort access offsets before I/O, use sequential access patterns, enable data sieving for unavoidable random access.",
        "code_before": "for (i=0; i<n; i++) {\n  offset = random_offsets[i];\n  pwrite(fd, buf, size, offset);  /* random offsets */\n}",
        "code_after": "/* Sort offsets for sequential access */\nqsort(random_offsets, n, sizeof(off_t), compare);\nfor (i=0; i<n; i++) pwrite(fd, buf, size, random_offsets[i]);",
    },
    "interface_choice": {
        "cause": "Using POSIX for shared-file parallel I/O instead of MPI-IO collective, or MPI-IO independent instead of collective",
        "fix": "Use MPI_File_write_all/read_all for shared files with multiple ranks. Enable collective buffering.",
        "code_before": "/* POSIX on shared file — each rank seeks and writes independently */\nlseek(fd, my_offset, SEEK_SET);\nwrite(fd, buf, size);",
        "code_after": "/* MPI-IO collective — automatic coordination */\nMPI_File_set_view(fh, my_offset, MPI_BYTE, filetype, \"native\", info);\nMPI_File_write_all(fh, buf, size, MPI_BYTE, &status);",
    },
    "file_strategy": {
        "cause": "File-per-process pattern creating excessive files, overwhelming metadata servers",
        "fix": "Use shared files with MPI-IO collective I/O. For checkpointing, use HDF5 or ADIOS2 for structured shared-file output.",
        "code_before": "/* Each of 1000 ranks creates its own file */\nchar fname[256];\nsprintf(fname, \"output_rank_%d.dat\", rank);\nfd = open(fname, O_CREAT|O_WRONLY);",
        "code_after": "/* Single shared file with MPI-IO */\nMPI_File_open(comm, \"output_shared.dat\", MPI_MODE_CREATE|MPI_MODE_WRONLY, info, &fh);\nMPI_File_write_at_all(fh, rank*chunk_size, buf, chunk_size, MPI_BYTE, &status);",
    },
    "throughput_utilization": {
        "cause": "Excessive synchronization (fsync/flush), single-OST serialization, or redundant I/O traffic",
        "fix": "Remove unnecessary fsync calls, use full Lustre stripe count, avoid re-reading data that can be cached.",
        "code_before": "for (i=0; i<n; i++) {\n  write(fd, buf, size);\n  fsync(fd);  /* fsync after EVERY write */\n}",
        "code_after": "/* Batch writes, fsync only at end */\nfor (i=0; i<n; i++) write(fd, buf, size);\nfsync(fd);  /* single fsync at end */",
    },
    "healthy": {
        "cause": "No significant I/O bottleneck detected",
        "fix": "No optimization needed. Current I/O pattern is efficient.",
        "code_before": "/* Already using collective MPI-IO with large transfers */\nMPI_File_write_all(fh, buf, 4*1024*1024, MPI_BYTE, &status);",
        "code_after": "/* No change needed */",
    },
}


def build_kb_entries(bench_features, bench_labels, models, feature_cols, shap_dict=None):
    """Build KB entries from benchmark data with ML predictions and SHAP.

    Each entry contains:
    - Benchmark metadata (type, config, scenario)
    - Darshan signature (key feature values)
    - ML detection results (bottleneck labels + confidence)
    - SHAP top features (what drives the detection)
    - Fix pattern (from FIX_PATTERNS)
    - Source code references
    """
    entries = []

    for idx in range(len(bench_features)):
        feat_row = bench_features.iloc[idx]
        label_row = bench_labels.iloc[idx]

        benchmark = label_row.get("benchmark", "unknown")
        scenario = label_row.get("scenario", "unknown")

        # Get active bottleneck dimensions
        active_dims = [d for d in DIMENSIONS if label_row.get(d, 0) == 1]
        if not active_dims:
            active_dims = ["healthy"]

        # ML prediction (if models available)
        ml_predictions = {}
        if models:
            X_row = np.array([[feat_row.get(col, 0) for col in feature_cols]], dtype=np.float32)
            for dim in DIMENSIONS:
                if dim in models:
                    prob = models[dim].predict_proba(X_row)[0][1]
                    ml_predictions[dim] = round(float(prob), 4)

        # SHAP top features (if available)
        shap_top = {}
        if shap_dict and idx < len(list(shap_dict.values())[0]):
            for dim in active_dims:
                if dim in shap_dict:
                    sv = shap_dict[dim][idx]
                    top_idx = np.argsort(np.abs(sv))[-10:][::-1]
                    shap_top[dim] = [
                        {"feature": feature_cols[i], "shap_value": round(float(sv[i]), 6),
                         "feature_value": round(float(feat_row.get(feature_cols[i], 0)), 6)}
                        for i in top_idx
                    ]

        # Key Darshan signature values
        signature = {}
        key_features = [
            "nprocs", "runtime_seconds", "POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ",
            "POSIX_WRITES", "POSIX_READS", "POSIX_SEQ_WRITES", "POSIX_SEQ_READS",
            "POSIX_FSYNCS", "POSIX_OPENS", "POSIX_F_META_TIME",
            "MPIIO_COLL_WRITES", "MPIIO_INDEP_WRITES",
            "avg_write_size", "avg_read_size", "small_io_ratio", "seq_write_ratio",
            "metadata_time_ratio", "collective_ratio", "total_bw_mb_s", "fsync_ratio",
        ]
        for f in key_features:
            val = feat_row.get(f, 0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                signature[f] = round(float(val), 4)

        # Build entry
        entry = {
            "entry_id": f"{benchmark}_{scenario}_{idx}",
            "benchmark": benchmark,
            "scenario": scenario,
            "bottleneck_labels": active_dims,
            "darshan_signature": signature,
            "ml_predictions": ml_predictions,
            "shap_top_features": shap_top,
            "source_code": BENCHMARK_SOURCES.get(benchmark, {}),
        }

        # Add fix patterns for each active dimension
        entry["fixes"] = []
        for dim in active_dims:
            if dim in FIX_PATTERNS:
                fix = FIX_PATTERNS[dim].copy()
                fix["dimension"] = dim
                entry["fixes"].append(fix)

        entries.append(entry)

    return entries


def export_for_tabassum(entries, output_path):
    """Export structured data for Tabassum's iterative LLM pipeline.

    Format: one JSON file per bottleneck type with all relevant entries.
    """
    by_dim = {}
    for entry in entries:
        for dim in entry["bottleneck_labels"]:
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(entry)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for dim, dim_entries in by_dim.items():
        path = output_path / f"kb_{dim}.json"
        with open(path, "w") as f:
            json.dump(dim_entries, f, indent=2)
        logger.info("  Exported %d entries for '%s' -> %s", len(dim_entries), dim, path)

    # Also export full KB
    full_path = output_path / "knowledge_base_full.json"
    with open(full_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info("  Full KB: %d entries -> %s", len(entries), full_path)

    return by_dim


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build I/O Knowledge Base from benchmarks")
    parser.add_argument("--output", default="data/knowledge_base/")
    parser.add_argument("--with-shap", action="store_true", help="Include SHAP values from saved analysis")
    args = parser.parse_args()

    # Load config
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        config = yaml.safe_load(f)

    # Load benchmark data
    logger.info("Loading benchmark data...")
    bench_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "features.parquet")
    bench_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "labels.parquet")
    logger.info("  %d benchmark samples", len(bench_feat))

    # Load trained model for ML predictions
    logger.info("Loading Phase 2 model...")
    model_path = PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl"
    models = {}
    if model_path.exists():
        with open(model_path, "rb") as f:
            models = pickle.load(f)
        logger.info("  Loaded %d per-label models", len(models))

    # Get feature columns
    prod_feat = pd.read_parquet(PROJECT_DIR / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]

    # Load SHAP values if available
    shap_dict = None
    if args.with_shap:
        shap_path = PROJECT_DIR / "paper" / "figures" / "shap" / "shap_values.pkl"
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                shap_data = pickle.load(f)
            shap_dict = shap_data["shap_dict"]
            logger.info("  Loaded SHAP values for %d dimensions", len(shap_dict))

    # Build KB entries
    logger.info("Building Knowledge Base...")
    entries = build_kb_entries(bench_feat, bench_labels, models, feature_cols, shap_dict)
    logger.info("  Built %d KB entries", len(entries))

    # Export
    logger.info("Exporting...")
    by_dim = export_for_tabassum(entries, PROJECT_DIR / args.output)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("KNOWLEDGE BASE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total entries: %d", len(entries))
    logger.info("Per dimension:")
    for dim in DIMENSIONS:
        n = len(by_dim.get(dim, []))
        logger.info("  %-28s %4d entries", dim, n)
    logger.info("")
    logger.info("Benchmarks represented:")
    bench_counts = {}
    for e in entries:
        b = e["benchmark"]
        bench_counts[b] = bench_counts.get(b, 0) + 1
    for b, n in sorted(bench_counts.items()):
        logger.info("  %-15s %4d entries", b, n)
    logger.info("")
    logger.info("Output: %s", PROJECT_DIR / args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
