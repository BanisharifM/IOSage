# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-03-24 21:50
**Abstract deadline**: Apr 1, 2026 (7 days)
**Paper deadline**: Apr 8, 2026 (14 days)
**AD deadline**: Apr 24, 2026 (30 days)

---

## Track Status

| Track | Description | Status | Key Results |
|-------|------------|--------|-------------|
| **Track A** | Export for Tabassum | DONE | exports/for_tabassum/ |
| **Track B** | Single-shot RAG+LLM (IOPrescriber) | DONE | 1.0 groundedness, 7.7x geomean |
| **Track C** | Iterative LLM optimization | CODE DONE, TESTING IN PROGRESS | SLURM test running |

---

## COMPLETED (Today 2026-03-24)

- [x] **4. MLP 5-seed training** -- 0.842 +/- 0.004 Micro-F1
- [x] **5. Missing metrics** -- Hamming=0.022, Subset Acc=0.869
- [x] **6. Paired bootstrap tests** -- all p<0.001, Cohen's d>3.6
- [x] **7. Majority class + threshold baselines** -- 0.158 and 0.298 Micro-F1
- [x] **11. Table 1: ML results** -- paper/tables/tab_ml_results.tex
- [x] **12. Table 2: Per-label** -- paper/tables/tab_per_label.tex
- [x] **13. Table 3: Baselines** -- paper/tables/tab_baselines.tex
- [x] **14. Table 4: LLM quality** -- paper/tables/tab_llm_quality.tex
- [x] **15. Table 5: Ablation** -- paper/tables/tab_ablation.tex
- [x] **16. Table 6: Closed-loop** -- paper/tables/tab_closed_loop.tex
- [x] **17. Speedup bar chart** -- paper/figures/fig_closed_loop_speedup.pdf
- [x] **18. Pipeline walkthrough** -- paper/figures/fig_pipeline_walkthrough.pdf
- [x] **19. LLM groundedness** -- paper/figures/fig_llm_groundedness.pdf
- [x] **20. Ablation chart** -- paper/figures/fig_ablation_trackb.pdf
- [x] **22. AD appendix** -- paper/appendix_ad.tex (4 pages, compiled PDF)
- [x] **29. Latency breakdown** -- ML p50=3.9ms, full cached p50=65.6ms
- [x] **30. Feature count verified** -- 186 engineered → 157 model features
- [x] Track C code: iterative_optimizer.py, benchmark_command_builder.py, iterative_executor.py
- [x] Track C config: configs/iterative.yaml (8 workloads, 3 models, 6 ablations)
- [x] Track C SLURM scripts: single, sweep, ablation
- [x] Track C figures script: scripts/generate_iterative_figures.py (5 figs + 4 tables)
- [x] Track C dry-run verified: LLM correctly proposes 64B→1MB fix

---

## IN PROGRESS

- [ ] **8. Track C real execution test** -- running from login node, SLURM benchmark jobs submitting
- [ ] **9-10. Track C convergence + comparison** -- waiting on execution results

---

## CRITICAL (Before Apr 1 Abstract)

- [ ] **1. Decide Track B vs Track C vs Both** -- user decision after Track C results
- [ ] **2. Write abstract** (250 words) -- currently empty in main.tex
- [ ] **3. System name** -- "IOSage" in paper vs "IOPrescriber" in code

---

## CRITICAL (Before Apr 8 Paper)

- [ ] **21. Paper writing** -- all 9 sections, ~6,500 words needed
- [ ] **Track C full sweep** -- sbatch scripts/run_iterative_sweep.slurm (after test passes)
- [ ] **Track C ablation** -- sbatch scripts/run_iterative_ablation.slurm (after test passes)

---

## IMPORTANT (Before Apr 24 AD)

- [ ] **23. Anonymous GitHub** -- create account at anonymous.4open.science
- [ ] **24. Zenodo upload** -- benchmark data + models (restricted link)
- [ ] **25. Reproduction notebooks** -- need ~5 more (have 3, need 8)

---

## NICE TO HAVE

- [ ] **26. More closed-loop pairs** -- mdtest metadata, DLIO checkpoint
- [ ] **27. Additional ML ablations** -- feature removal, GT-only, leave-one-out
- [ ] **28. IONavigator full 50-trace** -- 22/50 done

---

## Quick Reference: All Numbers for Paper

### ML Detection (436 GT test, XGBoost biquality w=100)
- Micro-F1: **0.923** [0.901, 0.943] 95% CI
- Macro-F1: **0.900** [0.863, 0.930] 95% CI
- Hamming loss: **0.022**
- Subset accuracy: **0.869**
- 5-seed: 0.920 +/- 0.004

### Baselines (same 436 GT test)
| System | Micro-F1 | Macro-F1 | Hamming | Subset |
|--------|----------|----------|---------|--------|
| XGBoost | **0.923** | **0.900** | **0.022** | **0.869** |
| LightGBM | 0.894 | 0.862 | 0.030 | 0.846 |
| RF | 0.894 | 0.880 | 0.030 | 0.819 |
| MLP | 0.842 | 0.787 | -- | -- |
| IONavigator | 0.500 | -- | -- | -- |
| Drishti | 0.384 | 0.283 | 0.221 | 0.305 |
| WisIO | 0.315 | 0.207 | 0.319 | 0.018 |
| Threshold | 0.298 | 0.202 | 0.255 | 0.064 |
| Majority | 0.158 | 0.036 | 0.226 | 0.170 |

### LLM (12 workloads x 3 models x 5 runs)
| Model | Groundedness | Latency |
|-------|-------------|---------|
| Claude | 1.000 | 14.2s |
| GPT-4o | 0.917 | 4.7s |
| Llama-70b | 1.000 | 13.3s |

### Closed-Loop (3 IOR pairs, write BW MiB/s)
| Pair | Before | After | Speedup |
|------|--------|-------|---------|
| Access gran. | 49 | 2,208 | 44.8x |
| Throughput | 500 | 3,408 | 6.8x |
| Access pattern | 2,212 | 3,281 | 1.5x |
| **Geomean** | | | **7.7x** |
| **Harmonic** | | | **3.6x** |

### Latency (p50 ms)
| Stage | p50 |
|-------|-----|
| Feature extraction | 55.6 |
| ML inference (8 dims) | 3.9 |
| SHAP explanation | 2.7 |
| KB retrieval | 2.8 |
| ML-only path | 59.4 |
| Full cached | 65.6 |
| Full with LLM | 14,000-20,000 |
