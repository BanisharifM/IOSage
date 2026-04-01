# Remaining Tasks Before SC 2026 Submission

**Last updated**: 2026-04-01 14:00
**System name**: IOSage
**Pipeline**: Darshan → ML detect → SHAP explain → KB retrieve → LLM recommend → validate via execution
**Paper deadline**: Apr 8, 2026 (7 days)
**AD deadline**: Apr 24, 2026 (23 days)

---

## IN PROGRESS

| Item | Status | Details |
|------|--------|---------|
| DLIO 3 workloads | **FIX APPLIED, RERUN NEEDED** | Root cause: `record_length_stdev` from unet3d_v100 preset (68M) not overridden when record_length changed to 64, causing 31 TiB array allocation. Fix: added `record_length_stdev=0` to all DLIO commands. Also fixed sacct polling bug (sub-step COMPLETED mistaken for job COMPLETED). First test job 17190758 confirmed data gen works. Need resubmit with polling fix. |

---

## NEXT STEPS (Priority Order)

### 1. Write the Paper (CRITICAL — 7 days, ~25 hours)
All materials ready in paper_materials.md Section 10. Related Work already complete.
- [ ] Abstract (250 words) — structure in abstract.tex comments
- [ ] Introduction (1.5 pages) — problem/gap/approach/contributions
- [ ] System Design (1.5 pages) — KB, ML, SHAP, closed-loop
- [ ] Dataset & Methodology (1.5 pages) — 1.37M logs, 131K cleaned, 623 GT
- [ ] Evaluation (2.5 pages) — ML detection, LLM recs, closed-loop, ablation, TraceBench
- [ ] Discussion (0.5 pages) — threats, limitations, biquality defense
- [ ] Conclusion (0.5 pages) — restate RQs, future work

### 2. DLIO Iterative Resubmit (after polling fix)
- [ ] Resubmit dlio_small_records with fixed polling
- [ ] Run dlio_shuffle_heavy and dlio_many_small_files
- [ ] Update paper_materials with DLIO results (will make 33/33 across 6 benchmarks)

### 3. Before Apr 24 (AD)
- [ ] Anonymous GitHub via anonymous.4open.science (30 min)
- [ ] Zenodo upload: benchmark logs + labels + models (1 hour)
- [ ] AD appendix final review (already compiled, review only)

---

## ALL COMPLETED EXPERIMENTS

| Item | Result | Source |
|------|--------|--------|
| ML detection (9 baselines) | 0.923 Micro-F1 [0.901, 0.943] | results/final_metrics.json |
| ML ablations (3 studies) | Feature +1.5%, biquality +3.6%, LOBO 6 benchmarks | results/ml_ablations.json |
| 5-seed multi-model | XGBoost 0.920±0.004, LightGBM 0.901±0.006, RF 0.897±0.004, MLP 0.842±0.004 | results/final_metrics.json |
| Weight sensitivity | w=100 on plateau (w=500 only +0.15%) | results/weight_sensitivity.json |
| Domain shift | Median KS=0.234, 88.5% features shifted | results/domain_shift_analysis.json |
| LLM evaluation (3 models, 180 runs) | Claude 1.000, Llama 1.000, GPT-4o 0.917 groundedness | evaluation_summary.json |
| IONavigator baseline (50 traces) | Micro-F1=0.419 (ours 2.2x better, 70K× faster) | ionavigator_baseline/ |
| TraceBench full (35 traces, 3-way) | IOSage P=0.774 (best), real-app F1=0.686 (best), IONav 110 FP | tracebench_full_evaluation.json |
| Production pipeline (4 Polaris jobs) | 100% ML accuracy, 100% groundedness | e2e_evaluation/ |
| E2E closed-loop Delta (64 procs) | 6.55x speedup, 3/4 rec categories align with expert fix | closed_loop/e2e_*.out |
| Closed-loop validation (7 pairs) | Geomean 4.26x, range 1.01x-44.8x | closed_loop_results.json |
| Iterative IOR+mdtest (24 runs, 3 LLMs) | Claude 5.90x, GPT-4o 4.18x, Llama 3.55x (6 shared) | trackc_*.json |
| Iterative HACC-IO (1 run) | 2.33x | trackc_hacc_*.json |
| Iterative h5bench (3 runs) | 578x, 55x, 6.25x | trackc_h5bench_*.json |
| Iterative custom (1 run) | Correctly classified healthy | trackc_custom_*.json |
| Iterative ablation (16/16) | no-ml high variance, no-kb lower, no-shap mixed, single-shot competitive | ablation_*.json |
| Pipeline ablation (5 conditions) | ML+KB essential (0→1.0 groundedness) | evaluation/ |
| Latency breakdown | ML 3.9ms, SHAP 2.7ms, KB 2.8ms, full cached 65.6ms | latency_breakdown.json |
| Novelty framing | "ML as Precision Gate" — 87% FP reduction, corroborated by OpenRCA/RCACopilot | novelty_framing.md |
| Related Work LaTeX | 530 words, 4 subsections, 12 refs, compiled | paper/sections/related_work.tex |
| Competitor analysis | 440 lines, 9 systems, AIIO 10 unstated limitations | Comprehensive_Competitor_Analysis.md |
| AD appendix | 4 pages, 5 contributions mapped, compiled | paper/appendix_ad.tex |
| LaTeX tables | 15 files (9 main + 4 iterative + 1 TraceBench + 1 ML ablations) | paper/tables/ |
| Figures | 39 PDFs + 67 PNGs, style-guide compliant | paper/figures/ |

## TERMINOLOGY (Clean)

| Old | New | Status |
|-----|-----|--------|
| Track B | Single-shot recommendation | Cleaned from ALL code, paper, docs |
| Track C | Iterative closed-loop | Cleaned from ALL code, paper, docs |
| IOPrescriber | IOSage | Cleaned from ALL code, paper, docs |
| trackb/trackc in filenames | Preserved (can't rename result files) | Annotated in paper_materials |
