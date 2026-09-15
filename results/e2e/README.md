# E2E NetCDF-4 write kernel on Delta (2026-09-14)

Kernels rebuilt against the current Cray stack (`benchmarks/e2e/Makefile`, `modules.sh`); run with
`scripts/run_e2e_variants.slurm` (4 nodes x 16 ranks, 8x8x1 decomposition, 32x32x128 local grid, 10
double variables, Darshan with `configs/darshan_runtime.conf`, 3 repeats per job, fresh file each).

| Variant | What it is | Wall time (s), 3 repeats | Median | Output |
|---|---|---|---|---|
| default (`write_3d_nc4`) | the kernel as written: collective NetCDF-4, implicit NC_FILL, default chunking. The community's shared "baseline" (Drishti PDSW'22, AIIO, ION). | 6.46, 5.60, 4.98 | 5.60 | 671,104,300 B |
| pathological | our constructed variant: NC_FILL + 1x1x64 chunks (+ nc_sync) | 34.79, 36.73, 30.96 | 34.79 | 744,549,314 B |
| ultra_optimized | published fixes: NC_NOFILL, process-matched chunks, 128 MB chunk cache, independent access | 2.66, 2.70, 2.19 | 2.66 | 671,135,554 B |

Speedups (ratio of medians): natural case default -> fixed **2.10x**; constructed case
pathological -> fixed **13.05x**. The default and fixed variants produce the same output
(work invariant); the pathological file is 11% larger (fill/chunk overhead).
Jobs 22079899 / 22079900 / 22079901; Darshan logs in `data/benchmark_logs/e2e/` (names carry the job id).
Earlier single-run figures (Mar 2026): 3.90 / 19.31 / 2.98 s.
