# Iterative closed-loop result files

Each `trackc_<workload>_<model>.json` file records one iterative optimization run
(baseline execution, per-iteration LLM recommendations, re-execution, measured
bandwidth). Files prefixed `ablation_` record the pipeline ablation conditions.

One workload required a re-execution:

- The original `ior_small_posix` Claude run had its baseline execution coincide
  with filesystem congestion on the shared Lustre scratch, which deflated the
  baseline bandwidth and inflated the computed speedup to 2240.93x. That run is
  preserved unmodified in `trackc_ior_small_posix_claude.json.orig`.
- The workload was re-executed on a quiescent filesystem (SLURM job 17380922),
  giving a corrected speedup of 5.97x. This is the value reported in the paper
  (Table VII, rounded to 6.0x).
- `trackc_ior_small_posix_claude.json` and
  `rerun_trackc_ior_small_posix_claude.json` both contain the corrected run, so
  scripts that aggregate `trackc_*.json` reproduce the paper values directly.
