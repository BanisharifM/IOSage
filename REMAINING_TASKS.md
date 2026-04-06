# Remaining Tasks — SC 2026 Submission

**Last updated**: 2026-04-06 07:25
**Deadline**: Apr 8, 2026
**Model**: Boost experiment (689 GT, 488 test)

---

## COMPLETED

| Item | Status | Paper location |
|------|--------|----------------|
| All paper numbers updated to boost (689 GT) | DONE | All sections |
| LLM quality table (Claude/GPT-4o/Llama) | DONE | Table VI: Claude 1.000, GPT-4o 0.917, Llama 0.917 |
| TraceBench full (per-app + cross-system) | DONE | Tables VIII, IX: IOSage P=0.774, F1=0.466 |
| Iterative closed-loop (51 runs, all 3 LLMs) | DONE | Table VII: geomeans 19.6x/6.3x/4.1x |
| Fair ablation (4 conditions) | DONE | Table V: C0 gnd=1.0, C1 gnd=1.0, C2 gnd=0.0, C5 gnd=0.0 |
| Production case study (50 jobs) | DONE | Eval text: 98% agreement, 1.000 groundedness |
| Facility stats (131K logs) | DONE | Eval text: 30.8%, 20.4%, 54.2% healthy |
| Per-benchmark F1 | DONE | Eval text |
| Threshold sweep | DONE | Eval text: F1 0.904--0.932 |
| Weight sensitivity | DONE | Fig (weight_sensitivity) |
| ML ablations (no-derived, GT-only, LOBO) | DONE | Eval text: 0.926, 0.908, per-bench drops |
| Latency breakdown | DONE | Latency table: 43ms detection path |
| t-SNE figure (Fig 7) | DONE | Regenerated with 689 benchmark samples |
| SHAP figure (Fig 4) | DONE | Regenerated from boost experiment SHAP values |
| All other figures (1-6) | DONE | Updated with boost data, improved readability |
| All tables | DONE | Updated with boost values, full column names, [!t] |
| Evaluation text numbers verified | DONE | LOBO, threshold, facility stats, Rec.P corrected |
| Paper compiled and pushed | DONE | 13 pages, synced with Overleaf |

## RUNNING (SLURM Jobs)

| Item | SLURM Job | Status | Purpose |
|------|-----------|--------|---------|
| Train LightGBM/RF/MLP (5 seeds each) | 17327514 | RUNNING | Fill missing Hamming/SubsetAcc in Table II |
| IOAgent with Claude | 17325123 | RUNNING | Fair comparison: IOAgent with same LLMs as IOSage |
| IOAgent with GPT-4o | 17325124 | RUNNING | Fair comparison |
| IOAgent with Llama-70b | 17325125 | RUNNING | Fair comparison |

## COMPLETED BUT NOT YET IN PAPER

| Item | Status | What's needed |
|------|--------|---------------|
| IOSage with gpt-4.1-mini | DONE (parse errors) | Summary script had key mismatch; need to verify if data is useful |
| Fair IOAgent comparison | RUNNING | When IOAgent jobs complete, add to Discussion section |

## PENDING (When SLURM jobs complete)

| Item | What to do |
|------|------------|
| ML comparison table (Table II) | Add Hamming/SubsetAcc for LightGBM and RF from job 17327514 |
| Fair IOAgent comparison | Add to Discussion: IOAgent performance with Claude/GPT-4o/Llama |
| E2E benchmark | Verify 6.5x speedup with new model (or note unchanged) |

---

## NOTES

- All iterative closed-loop: 51/51 successful (17 workloads × 3 LLMs)
- IOSage gpt-4.1-mini had 12 parse errors — summary script expects step4 keys but pipeline returns step3 (SHAP removed). Raw data may still be valid.
- MLP model from previous experiment performed poorly (0.303 F1) — was trained on wrong feature set. New training job retrains properly with same biquality approach.
- Paper latency table uses old KB size (623 entries) but this doesn't affect timing.
