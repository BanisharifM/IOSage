# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-04-01
**System name**: IOSage
**Pipeline**: Darshan → ML detect → SHAP explain → KB retrieve → LLM recommend → validate via execution
**Abstract deadline**: Apr 1, 2026 (TODAY)
**Paper deadline**: Apr 8, 2026 (7 days)
**AD deadline**: Apr 24, 2026 (23 days)

---

## IOSage: ONE Unified System

**4 components, one pipeline:**
1. ML multi-label classifier (0.923 F1, 8 bottleneck dimensions)
2. SHAP per-label feature attribution
3. KB-grounded LLM code-level recommendations (100% groundedness)
4. Closed-loop validator (executes recommendations, measures speedup)

Component (4) is the **evaluation methodology**, not a separate feature.
The paper presents ONE system, NOT two tracks.

---

## COMPLETED EXPERIMENTS

| Item | Result | Files |
|------|--------|-------|
| ML detection (9 baselines, 5 seeds) | 0.923 Micro-F1 | results/final_metrics.json |
| ML ablations (3 studies) | Feature +1.5%, biquality +3.6%, LOBO | results/ml_ablations.json |
| Per-benchmark F1 | 0.875-1.000 across all 6 | results/per_benchmark_f1.json |
| Weight sensitivity | w=100 confirmed optimal | results/weight_sensitivity.json |
| Domain shift | Median KS=0.234, explainable | results/domain_shift_analysis.json |
| LLM evaluation (3 models) | 1.0/0.917/1.0 groundedness | results/llm_evaluation/ |
| Groundedness atomic | 88-98% claim-level | results/groundedness_atomic.json |
| IONavigator baseline | 50/50, Micro-F1=0.419 | results/ionavigator_baseline/ |
| TraceBench 3-way (KEY NOVELTY) | F1=0.649, 87% FP reduction | results/e2e_evaluation/tracebench_comprehensive.json |
| Production case study (50 logs) | 100% groundedness | results/production_case_study/ |
| E2E production pipeline (4 jobs) | 100% ML accuracy | results/e2e_evaluation/e2e_full_pipeline.json |
| Closed-loop validation (7 pairs) | 1.01x-44.8x speedup | results/closed_loop/ |
| Iterative validation (24 runs) | Claude 5.90x geomean | results/iterative/trackc_*.json |
| Pipeline ablation (5 conditions) | ML+KB both essential | results/ablation/ |
| Novelty framing | "ML as Precision Gate" | docs/1_strategy/novelty_framing.md |
| AD appendix | Created + compiled | paper/appendix_ad.tex |
| LaTeX tables | 13 files | paper/tables/ |
| Figures | 19 PDFs | paper/figures/ |

---

## IN PROGRESS (Running Now)

| Item | Status | Check |
|------|--------|-------|
| Iterative validation ablation reruns | nohup running | `tail -f results/iterative/ablation_rerun_v3.log` |
| E2E pathological on Delta | SLURM 17178076 | `sacct -j 17178076 --format=State,Elapsed` |
| New benchmark validation (HACC-IO, h5bench, custom) | nohup running | `tail -f results/iterative/trackc_new_benchmarks.log` |

**When E2E pathological completes:**
```bash
source .env && python scripts/run_e2e_pipeline_analysis.py
```

---

## CRITICAL — TODAY Apr 1

- [ ] **Write abstract** (250 words)
- [ ] **System name confirmed**: IOSage

## CRITICAL — Before Apr 8

- [ ] **Write paper** (all sections, ~6500 words)
  - Introduction (1p): ML as Precision Gate, 87% FP reduction
  - Related Work (1p): AIIO/IOAgent/STELLAR/WisIO/Drishti/RCACopilot
  - System Design (2p): 4 components, precision gate concept, KB construction
  - Evaluation (4.5p): ML detection + 3-way TraceBench + groundedness + closed-loop + production + ablation
  - Discussion (1p): limitations, domain shift, benchmark-only validation
  - Conclusion (0.5p)
- [ ] **Regenerate iterative figures** with final ablation data
- [ ] **Run h5bench + HACC-IO through LLM** (Tabassum handles mdtest)
- [ ] **Commit all uncommitted changes**

## IMPORTANT — Before Apr 24

- [ ] Anonymous GitHub (anonymous.4open.science)
- [ ] Zenodo upload (benchmark data + models)
- [ ] AD appendix final review
- [ ] Reproduction notebooks

---

## Quick Reference: All Numbers

### ML Detection
- Micro-F1: **0.923** [0.901, 0.943], Macro-F1: **0.900**, Hamming: **0.022**, Subset Acc: **0.869**

### Precision Gate Effect (KEY NOVELTY — 3-way on same 9 traces)
| System | Precision | F1 | FP |
|--------|-----------|-----|-----|
| **IOSage (ML+LLM)** | **0.750** | **0.649** | **4** |
| Drishti (rules) | 0.667 | 0.615 | 6 |
| IONavigator (LLM-only) | 0.326 | 0.448 | **31** |

### All Baselines (436 GT test)
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

### LLM Recommendation Quality
| Model | Groundedness (citation) | Groundedness (claim) | Latency |
|-------|------------------------|---------------------|---------|
| Claude | 1.000 | 0.906 | 14.2s |
| GPT-4o | 1.000 | 0.983 | 4.7s |
| Llama-70b | 1.000 | 0.886 | 13.3s |

### Closed-Loop Speedup (7 pairs)
| Pair | Speedup |
|------|---------|
| IOR small_posix | 44.8x |
| IOR small_direct (O_DIRECT) | 11.9x |
| IOR fsync_heavy | 6.8x |
| IOR collective (64 ranks) | 2.05x |
| IOR random_to_seq | 1.5x |
| mdtest metadata_storm | 1.05x |
| mdtest fpp_explosion | 1.01x |

### Iterative Validation (24 runs, 8 workloads × 3 LLMs)
| Model | Geomean | Avg Iters | Cost |
|-------|---------|-----------|------|
| Claude | 5.90x | 1.1 | $0.10 |
| GPT-4o | 4.18x | 2.1 | $0.09 |
| Llama | 3.55x | 2.0 | $0.02 |

### Production Validation
- 4 Polaris production jobs: 100% ML accuracy, 100% groundedness
- 50 random production logs: 100% groundedness, distribution matches population

### Latency
- ML-only: 59.4ms (p50), Full with LLM: 14-20s
