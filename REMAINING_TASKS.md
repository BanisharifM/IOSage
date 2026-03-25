# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-03-25 02:00
**Abstract deadline**: Apr 1, 2026 (7 days)
**Paper deadline**: Apr 8, 2026 (14 days)
**AD deadline**: Apr 24, 2026 (30 days)

---

## Track Status

| Track | Description | Status |
|-------|------------|--------|
| Track A | Export for Tabassum | DONE |
| Track B | Single-shot RAG+LLM (IOPrescriber) | DONE |
| Track C | Iterative LLM optimization | CODE DONE, SLURM BLOCKED (6K queue) |

---

## Review Findings Addressed (Since Tough Review)

| Finding | Action | Result |
|---------|--------|--------|
| W3: No real app validation | Ran 4 Polaris production jobs through full pipeline | 100% accuracy, 100% groundedness |
| W7: IONavigator incomplete | Completed all 50 traces (was 22) | Micro-F1=0.419 (our 0.923 = 2.2x better) |
| W9: Domain shift uncharacterized | t-SNE + KS tests on 157 features | Median KS=0.234, shift explainable |
| W12: No weight sensitivity | Tested w={1,10,50,100,200,500} | w=100 confirmed near-optimal (plateau) |
| W4 partial: Per-label CIs | Bootstrap CIs in final_metrics.json | All per-label CIs computed |
| W13: MLP underperformance | 5-seed training completed | 0.842 +/- 0.004 (trees expected to win) |

---

## CRITICAL — Before Apr 1 (Abstract)

- [ ] **Write abstract** (250 words) — currently empty in main.tex
- [ ] **Decide system name** — "IOSage" in paper vs "IOPrescriber" in code (pick one)
- [ ] **Decide Track B vs C vs Both** — Track C needs SLURM results; decision by Mar 30

## CRITICAL — Before Apr 8 (Paper)

### Paper Writing (W1 — rejection reason if not done)
- [ ] Write Introduction (~1.5 pages): gap statement, contributions (C1-C5), pipeline overview
- [ ] Write Related Work (~1 page): AIIO/IOAgent/STELLAR/WisIO/Drishti/RCACopilot, our positioning
- [ ] Write System Design (~1.5 pages): architecture, ML, SHAP, KB, LLM, closed-loop
- [ ] Write Dataset & Methodology (~1.5 pages): 1.37M logs, cleaning, features, biquality, GT construction
- [ ] Write Evaluation (~2.5 pages): ML detection, baselines, ablations, LLM quality, closed-loop, E2E
- [ ] Write Discussion (~0.5 pages): threats to validity, limitations, domain shift
- [ ] Write Conclusion (~0.5 pages): RQ answers, future work

### Closed-Loop Expansion (W2 — 2nd most likely rejection reason)
- [ ] Run 4 extended closed-loop pairs when SLURM clears — scripts/run_closed_loop_extended.slurm
  - mdtest_metadata_storm (metadata_intensity)
  - mdtest_fpp_explosion (file_strategy)
  - ior_collective_vs_independent (interface_choice, 64 ranks / 4 nodes)
  - ior_small_to_large_direct (access_granularity, O_DIRECT)
- [ ] Target: 7 validated pairs across IOR + mdtest (currently 3 IOR-only)

### Track C Execution (W14)
- [ ] Run Track C test when SLURM clears: `source .env && python -m src.llm.iterative_optimizer --workload ior_small_posix --model claude-sonnet --max-iterations 5 --n-runs 1 --output results/iterative/test_ior_small_posix_real.json`
- [ ] If successful, run full sweep: `sbatch scripts/run_iterative_sweep.slurm`
- [ ] If successful, run ablation: `sbatch scripts/run_iterative_ablation.slurm`

### Novelty Framing (W6 — 1st most likely rejection reason)
- [ ] Frame as "validated end-to-end system" not "new algorithms" (in introduction)
- [ ] Emphasize what neither AIIO nor IOAgent achieves alone: closed-loop with measured speedup
- [ ] Cite RCACopilot/STELLAR as precedent for systems contribution papers at top venues
- [ ] Ablation showing every component is necessary (ML, KB, SHAP all contribute)

### TraceBench Decision (W5)
- [ ] Decide: include TraceBench 0.103 with honest framing, or exclude?
  - Review says including is better (avoids "cherry-picking" accusation if reviewer finds it in artifact)
  - If including: frame as taxonomy mismatch discussion, not as failure
  - If excluding: remove from artifact or explain clearly in AD

### Paper Space Management (D.2)
- [ ] Ruthless space allocation: max 10 pages
- [ ] Move to supplementary: dataset EDA details, full per-label breakdowns, LOBO details, latency
- [ ] Keep in paper: architecture, biquality, ML results + baselines, LLM quality, closed-loop, E2E

## IMPORTANT — Before Apr 24 (AD Appendix)

- [ ] **Anonymous GitHub** — create at anonymous.4open.science, push code
- [ ] **Zenodo upload** — benchmark data + models (restricted link)
- [ ] **Reproduction notebooks** — need ~5 more (have 3, need 8)
- [ ] **AD appendix review** — appendix_ad.tex exists, verify it matches final paper content

## NICE TO HAVE (Strengthens Paper)

- [ ] **DLIO closed-loop pair** — checkpoint bottleneck (not yet configured)
- [ ] **More E2E production cases** — run pipeline on 10+ production logs with expert validation
- [ ] **MLP per-label analysis** — where exactly does MLP fail vs trees? (addresses W13 deeper)
- [ ] **Groundedness atomic verification** — verify each claim matches KB entry, not just citation (W8)
- [ ] **"Healthy" label framing** — present as "7 dimensions + 1 health status" (W10)
- [ ] **Cleanlab downgrade** — frame as sanity check, not contribution (W11)

---

## Quick Reference: All Numbers for Paper

### ML Detection (436 GT test, XGBoost biquality w=100)
- Micro-F1: **0.923** [0.901, 0.943] 95% CI
- Macro-F1: **0.900** [0.863, 0.930] 95% CI
- Hamming loss: **0.022**
- Subset accuracy: **0.869**
- 5-seed: 0.920 +/- 0.004

### All Baselines (same 436 GT test)
| System | Micro-F1 | Macro-F1 | Type |
|--------|----------|----------|------|
| **XGBoost (ours)** | **0.923** | **0.900** | ML |
| LightGBM | 0.894 | 0.862 | ML |
| RF | 0.894 | 0.880 | ML |
| MLP | 0.842 | 0.787 | DL |
| IONavigator | 0.419 | 0.298 | LLM (50 traces) |
| Drishti | 0.384 | 0.283 | Rules |
| WisIO | 0.315 | 0.207 | Rules |
| Threshold | 0.298 | 0.202 | Statistical |
| Majority | 0.158 | 0.036 | Trivial |

### LLM Quality (12 workloads x 3 models x 5 runs)
| Model | Groundedness | Latency |
|-------|-------------|---------|
| Claude | 1.000 | 14.2s |
| GPT-4o | 0.917 | 4.7s |
| Llama-70b | 1.000 | 13.3s |

### Closed-Loop (3 IOR pairs, write BW MiB/s)
| Pair | Before | After | Speedup |
|------|--------|-------|---------|
| Access granularity | 49 | 2,208 | 44.8x |
| Throughput (fsync) | 500 | 3,408 | 6.8x |
| Access pattern | 2,212 | 3,281 | 1.5x |
| **Geomean** | | | **7.7x** |

### E2E Production Validation (4 Polaris jobs)
- ML accuracy: 4/4 (100%), 5 TP, 0 FP, 0 FN
- LLM recommendations: 9 total, 100% grounded
- Healthy job correctly identified (no false recs)

### ML Ablations
- Derived features: +1.5% (0.908→0.923)
- Biquality vs GT-only: +3.6% (0.887→0.923)
- LOBO: custom→0.0, mdtest→0.344 (each benchmark essential)

### Weight Sensitivity
- w=100 near-optimal (plateau at w>=50)
- Range: 0.859 (w=1) to 0.924 (w=500)

### Domain Shift
- Median KS: 0.234 (moderate)
- 88.5% features show significant shift
- Top shifted: has_apmpi, io_active_fraction, STDIO counters (explainable)

### Latency (p50 ms)
- ML-only: 59.4ms, Full cached: 65.6ms, Full with LLM: 14-20s
