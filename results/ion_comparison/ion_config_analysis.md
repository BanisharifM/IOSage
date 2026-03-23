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

| Aspect | Drishti | ION | IOPrescriber |
|--------|--------|-----|-------------|
| Detection method | 30 threshold rules | LLM in-context learning | ML (0.920 F1) + SHAP + LLM |
| Input | Darshan log file | Darshan CSV text | Darshan features (157 dims) |
| Speed | <1s per job | Seconds per job (LLM) | <1ms (ML) + 14s (LLM optional) |
| Output | Issue flags (HIGH/WARN) | Text diagnosis + code templates | Grounded recommendations + code fixes |
| ML component | None | None | XGBoost multi-label, 5 seeds |
| Benchmark KB | None | None | 623 entries with source code |
| Closed-loop | None | None | Measured speedup (39x IOR, 7x interface) |
| Reproducibility | Open source (pip) | No public code | Anonymous GitHub + LLM cache |

**How ION compared with Drishti (from paper slides):**
ION showed side-by-side text outputs on E2E application:
  - Drishti: terse threshold flags ("99.81% misaligned", "99.90% load imbalance")
  - ION: verbose LLM text with reasoning and context
  - Both detect same issues; ION adds explanation
ION reports NO quantitative metrics — evaluation is entirely qualitative.

**ION public data:** github.com/cegersdoerfer/io500-trace-ds (14 configs with GT labels)
**ION CLI:** github.com/cegersdoerfer/ION-cli (web API wrapper, not the analysis code)
**ION code:** NOT open source (runs on AWS EC2 web service)

**We ran Drishti on these same IO500 configs (3-way comparison, same data):**
  Config1 (2KB): Drishti=access_gran+metadata, ML=same
  Config2 (1MB): Drishti=access_gran, ML=access_gran+throughput
  Config3 (FPP): Both=access_gran
  Config5 (4KB): Both=throughput

## Key Advantage
ION gives generic advice: "consider using larger I/O buffers"
IOPrescriber gives specific, grounded advice: "change write(fd, buf, 64) to
MPI_File_write_all(fh, buf, 1048576) — benchmark evidence shows 39x speedup
(entry: ior_small_posix_t64_n4_r1_0)"
