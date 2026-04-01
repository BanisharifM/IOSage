# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-04-01 04:30
**System name**: IOSage
**Pipeline**: Darshan → ML detect → SHAP explain → KB retrieve → LLM recommend → validate via execution
**Paper deadline**: Apr 8, 2026 (7 days)
**AD deadline**: Apr 24, 2026 (23 days)

---

## IN PROGRESS

| Item | Status | Check |
|------|--------|-------|
| DLIO 3 workloads (GPU rerun) | Running (PID 1467772) | `tail -f results/iterative/dlio_final_runs.log` |

---

## NEXT STEPS (Priority Order)

### 1. Write the Paper (CRITICAL — 7 days)
- [ ] Write Abstract (250 words)
- [ ] Write Introduction (1p)
- [ ] Write System Design (2p)
- [ ] Write Dataset & Methodology (1.5p)
- [ ] Write Evaluation (4.5p)
- [ ] Write Discussion (1p)
- [ ] Write Conclusion (0.5p)

### 2. When DLIO finishes
- [ ] Update paper_materials with DLIO results
- [ ] Regenerate iterative figures with all 6 benchmarks

### 3. Before Apr 24 (AD)
- [ ] Anonymous GitHub
- [ ] Zenodo upload
- [ ] AD appendix final review

---

## ALL COMPLETED EXPERIMENTS

| Item | Result |
|------|--------|
| ML detection (9 baselines) | 0.923 Micro-F1 |
| ML ablations (3 studies) | Feature +1.5%, biquality +3.6%, LOBO |
| Per-benchmark F1 | 0.875-1.000 (all 6 benchmarks) |
| Weight sensitivity | w=100 confirmed |
| Domain shift | Median KS=0.234 |
| LLM evaluation (3 models) | 1.0/0.917/1.0 groundedness |
| Groundedness atomic | 88-98% claim-level |
| IONavigator baseline | 50/50, Micro-F1=0.419 |
| TraceBench 3-way (KEY NOVELTY) | F1=0.649, 87% FP reduction |
| Production case study (50 logs) | 100% groundedness |
| E2E production pipeline (4 jobs) | 100% ML accuracy |
| E2E closed-loop Delta | 6.55x speedup, 3/4 rec alignment |
| Closed-loop validation (7 pairs) | 1.01x-44.8x |
| Iterative IOR (21 runs, 3 LLMs) | Claude 5.90x geomean |
| Iterative mdtest (3 runs) | Claude 9.42x |
| Iterative HACC-IO (2 runs) | 2.33x |
| Iterative h5bench (3 runs) | 578x, 55x, 6.25x |
| Iterative custom (1 run) | Correctly healthy |
| Iterative ablation (16/16) | All conditions complete |
| Novelty framing | "ML as Precision Gate" |
| Related Work LaTeX | 546 words, 12 refs, compiled |
| Competitor analysis | 440 lines, 9 systems |
| AD appendix | Created + compiled |
| LaTeX tables | 13 files |
| Figures | 19 PDFs |
