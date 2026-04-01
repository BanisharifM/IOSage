# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-04-01
**Abstract deadline**: Apr 1, 2026 (TODAY)
**Paper deadline**: Apr 8, 2026 (7 days)
**AD deadline**: Apr 24, 2026 (23 days)

---

## COMPLETED EXPERIMENTS

| Item | Result | Files |
|------|--------|-------|
| ML detection (9 baselines, 5 seeds) | 0.923 Micro-F1 | results/final_metrics.json |
| ML ablations (3 studies) | Feature +1.5%, biquality +3.6%, LOBO | results/ml_ablations.json |
| Per-benchmark F1 | 0.875-1.000 across all 6 | results/per_benchmark_f1.json |
| Weight sensitivity | w=100 confirmed optimal | results/weight_sensitivity.json |
| Domain shift | Median KS=0.234, explainable | results/domain_shift_analysis.json |
| LLM evaluation (Track B) | 1.0/0.917/1.0 groundedness | results/llm_evaluation/ |
| Groundedness atomic | 88-98% claim-level | results/groundedness_atomic.json |
| IONavigator baseline | 50/50, Micro-F1=0.419 | results/ionavigator_baseline/ |
| TraceBench 3-way comparison | F1=0.649, 87% FP reduction | results/e2e_evaluation/tracebench_comprehensive.json |
| Production case study (50 logs) | 100% groundedness | results/production_case_study/ |
| E2E production pipeline (4 jobs) | 100% ML accuracy | results/e2e_evaluation/e2e_full_pipeline.json |
| Track B closed-loop (3 original) | 44.8x, 6.8x, 1.5x | results/closed_loop/closed_loop_results.json |
| Track B extended closed-loop (4 new) | 2.05x, 11.9x, 1.05x, 1.01x | results/closed_loop/extended_*.out |
| Track C main runs (24/24) | Claude 5.90x, GPT-4o 4.18x, Llama 3.55x | results/iterative/trackc_*.json |
| Novelty framing research | "ML as Precision Gate" | docs/1_strategy/novelty_framing.md |
| AD appendix | Created + compiled | paper/appendix_ad.tex |
| LaTeX tables | 13 files | paper/tables/ |
| Figures | 19 PDFs | paper/figures/ |

---

## IN PROGRESS (Running Now)

| Item | Status | How to Check |
|------|--------|-------------|
| Track C ablation reruns (6 failed) | nohup PID 3505787 | `tail -f results/iterative/ablation_rerun_v2.log` |
| E2E pathological on Delta | SLURM 17177584 (48h) | `sacct -j 17177584 --format=State,Elapsed` |
| E2E ultra-optimized on Delta | SLURM 17177585 (1h) | `sacct -j 17177585 --format=State,Elapsed` |
| E2E baseline on Delta | SLURM 17177586 (2h) | `sacct -j 17177586 --format=State,Elapsed` |

**When E2E jobs complete → Run pipeline analysis:**
```bash
source .env && python scripts/run_e2e_pipeline_analysis.py
```
This will parse Darshan logs, run ML+SHAP+KB+LLM, and compute speedups.

---

## CRITICAL — TODAY Apr 1

- [ ] **Write abstract** (250 words)
- [ ] **Decide system name** — IOSage or IOPrescriber

## CRITICAL — Before Apr 8

- [ ] **Write paper** (all 9 sections, ~6500 words)
- [ ] **Run E2E pipeline analysis** (after SLURM jobs complete)
- [ ] **Regenerate Track C tables/figures** with final ablation data (after reruns complete)
- [ ] **Run h5bench + HACC-IO through LLM** (you handle these, Tabassum handles mdtest)
- [ ] **Commit all uncommitted changes**

## IMPORTANT — Before Apr 24

- [ ] Anonymous GitHub (anonymous.4open.science)
- [ ] Zenodo upload (benchmark data + models)
- [ ] AD appendix final review
- [ ] Reproduction notebooks (~5 more needed)

---

## Quick Reference: All Numbers

### ML Detection
- Micro-F1: **0.923** [0.901, 0.943], Macro-F1: **0.900**, Hamming: **0.022**, Subset Acc: **0.869**

### Baselines (436 GT test)
| System | Micro-F1 | Type |
|--------|----------|------|
| XGBoost (ours) | **0.923** | ML |
| LightGBM | 0.894 | ML |
| RF | 0.894 | ML |
| MLP | 0.842 | DL |
| IONavigator | 0.419 | LLM (50 traces) |
| Drishti | 0.384 | Rules |
| WisIO | 0.315 | Rules |
| Threshold | 0.298 | Statistical |
| Majority | 0.158 | Trivial |

### Track B Closed-Loop (7 pairs)
| Pair | Speedup |
|------|---------|
| IOR small_posix | 44.8x |
| IOR small_direct (O_DIRECT) | 11.9x |
| IOR fsync_heavy | 6.8x |
| IOR collective (64 ranks) | 2.05x |
| IOR random_to_seq | 1.5x |
| mdtest metadata_storm | 1.05x |
| mdtest fpp_explosion | 1.01x |

### Track C Iterative (24 runs)
| Model | Geomean | Avg Iters | Cost |
|-------|---------|-----------|------|
| Claude | 5.90x | 1.1 | $0.10 |
| GPT-4o | 4.18x | 2.1 | $0.09 |
| Llama | 3.55x | 2.0 | $0.02 |

### TraceBench 3-Way (KEY NOVELTY)
| System | Precision | F1 | FP |
|--------|-----------|-----|-----|
| Ours | 0.750 | 0.649 | 4 |
| Drishti | 0.667 | 0.615 | 6 |
| IONavigator | 0.326 | 0.448 | 31 |

### Groundedness
- Citation: 1.0 (Claude/Llama), 0.917 (GPT-4o)
- Claim-level: 0.906 (Claude), 0.983 (GPT-4o), 0.886 (Llama)
