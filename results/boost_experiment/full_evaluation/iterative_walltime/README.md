# Iterative closed-loop results recomputed under a wall-time objective

Produced by `scripts/recompute_iterative_walltime.py` on 2026-09-09 from the
retained artifacts of the April 2026 runs. No benchmark was re-executed.

The canonical histories in `../iterative/trackc_*.json` are unchanged. Each
`walltime_<workload>_<model>.json` here is derived from the matching
`trackc_` file, the SLURM stdout in `results/iterative/`, and the Darshan
logs in `data/benchmark_logs/iterative/`. Every baseline and iteration was
located by reproducing the bandwidth value the optimizer recorded, so the
mapping from history entry to SLURM job is exact (`bw_match_rel_err` = 0).

## Metric definitions

| column | definition |
|---|---|
| `paper_bw_speedup` | `write_bw_mb_s` ratio from the Darshan log the executor selected (newest by mtime). This is what Table VII of the SC26 submission reports. |
| `walltime_at_paper_best` | Darshan `run_time` ratio, baseline over the configuration the loop chose. Multi-phase jobs (h5bench write then read, DLIO datagen then training) sum sequential phases and take the longest concurrent rank within a phase. |
| `tool_at_paper_best` | The benchmark's own stdout figure where one exists: IOR Max Write, mdtest file-creation rate, HACC-IO write bandwidth, h5bench observed write completion time. |
| `guarded_walltime` | `walltime_at_paper_best` if the work invariant holds, else 1.0 (fix rejected, baseline kept). |
| `work_changed` | true when total bytes moved by the chosen configuration differ from the baseline by more than 25 percent. Bytes come from Darshan for IOR, HACC-IO and custom, and from the benchmark configuration for mdtest and DLIO (see below). |
| `best_iter_agrees` | whether the iteration with the best wall time is the one the bandwidth-driven loop selected. |
| `n_decision_disagreements` | iterations where the executor's regression rule (`speedup < 0.9`) gives a different keep/rollback answer under the two metrics. |

`geomeans.json` gives per-model geometric means over the 12 workloads the
classifier did not exclude as healthy, under each definition, plus the
subset the submitted table used (bandwidth speedup above 1.0).

## Two measurement caveats that the recompute exposed

1. **Darshan record cap.** The IOR/mdtest SLURM path leaves Darshan's
   default `MAX_RECORDS` (1,024 file records per module per rank). Any job
   touching more files than that is truncated: the mdtest baseline that
   created 160,000 files reports 16 x 1,024 = 16,384 opens, writes and
   bytes. Bandwidth computed from those counters is not meaningful, and
   Darshan byte counts cannot serve as the work invariant for such jobs.
   `record_cap_hit` marks affected rows; for mdtest the work check uses
   `items_per_rank x write_bytes x ranks` from the configuration instead.
   The same cap affects 102 of the 689 DIOBench ground-truth logs (all 93
   mdtest metadata_intensity samples and 9 mdtest file_strategy samples),
   whose file-count features saturate at 1,025 per rank regardless of the
   configured item count.

2. **h5bench phase selection.** The executor picks the newest Darshan log
   by mtime, which for h5bench is the read-phase log. The submitted
   h5bench bandwidth speedups therefore compare read-phase total
   bandwidth on sub-second runs, while every proposed change targeted the
   write phase. `tool_at_paper_best` for h5bench uses the benchmark's own
   write-phase completion time and is the figure to trust.

## Reading the summary

- IOR (18 runs): wall time, Darshan bandwidth and the IOR-reported figure
  agree in direction and within roughly 40 percent; bytes are identical
  (ratio 1.00 to 1.05). These rows stand under either metric.
- mdtest: wall-time gains are real for Claude (138x) and GPT-4.1-mini
  (9.6x) with the same bytes written into 100x and 10x fewer files; the
  Llama configuration wrote 100x fewer bytes and is rejected by the guard.
- h5bench: six of eight submitted values do not survive the write-phase
  measurement; Claude on small_access (11.5x wall, 26x by the tool's own
  timer) and Llama on indep_vs_coll (4.2x) do.
- DLIO: wall-time gains are 1.0x to 2.4x; eight of twelve chosen
  configurations changed the training-set bytes (record length or file
  count) and are rejected by the guard.
- HACC-IO: 0.8x to 2.0x under every metric.
- The bandwidth-driven and wall-time-driven objectives pick the same best
  iteration in 25 of 48 runs, and 24 of 48 runs contain at least one
  keep/rollback decision that flips between the metrics. The wall-time
  values here are therefore the wall time of the configurations a
  bandwidth-driven search chose, not the outcome of a wall-time-driven
  search.
