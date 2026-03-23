# ION Paper IO500 Configuration Analysis

## Source
ION: Luettgau et al., "Navigating the HPC I/O Optimization Journey using LLMs"
HotStorage 2024: https://dl.acm.org/doi/10.1145/3655038.3665950

## Our Analysis on the Same IO500 Configs

These are the exact configs from the ION paper, run on Delta (4 nodes, 256 tasks).
Darshan logs collected and analyzed through IOPrescriber.

| Config | Description | IOPrescriber Detection | BW (MB/s) | Avg Write |
|--------|------------|----------------------|-----------|-----------|
| Config1 | Small I/O (2KB, shared file) | access_granularity (1.0), metadata (0.82) | 7.4 | 2KB |
| Config2 | Aligned I/O (1MB, shared file) | access_granularity (1.0), throughput (0.55) | 17.9 | 989KB |
| Config3 | Optimized (1MB, file-per-proc) | access_granularity (1.0) | 41.4 | 989KB |
| Config5 | 4KB Random I/O | throughput_utilization (0.99) | 0.0 | 108B |

## Comparison: ION vs IOPrescriber

| Aspect | ION | IOPrescriber |
|--------|-----|-------------|
| Detection method | LLM in-context learning | ML classifier (0.920 F1) + SHAP |
| Input | Darshan CSV text | Darshan features (157 dimensions) |
| Speed | Seconds per job (LLM inference) | <1ms per job (ML) + optional LLM |
| Recommendations | Generic static code templates | Grounded in benchmark KB with measured speedup |
| Evaluation | Qualitative | Quantitative (groundedness=1.0, speedup=39x) |
| ML component | None | XGBoost multi-label, 5 seeds |
| Benchmark KB | None | 623 entries with source code |
| Closed-loop | None | Measured speedup on IOR + E2E |
| Reproducibility | No public code | Anonymous GitHub + LLM cache |

## Key Advantage
ION gives generic advice: "consider using larger I/O buffers"
IOPrescriber gives specific, grounded advice: "change write(fd, buf, 64) to
MPI_File_write_all(fh, buf, 1048576) — benchmark evidence shows 39x speedup
(entry: ior_small_posix_t64_n4_r1_0)"
