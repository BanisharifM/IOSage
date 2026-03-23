# SC 2026 Research Quality Checklist

> Mandatory supplementary document. Submissions without this are desk rejected.
> Does NOT count toward the 10-page limit.

---

## 1. Experimental Methodology

**Q: Are all experimental parameters fully specified?**
Yes. All hyperparameters documented in configs/training.yaml. ML models: XGBoost (n_estimators=500, max_depth=6, learning_rate=0.05), LightGBM (n_estimators=500, num_leaves=63), Random Forest (n_estimators=500, max_depth=15), MLP (256-128-64, ReLU, 500 epochs). Biquality training weight=100 for ground-truth samples. LLM: temperature=0, max_tokens=2000. All random seeds: [42, 123, 456, 789, 1024].

**Q: Are baselines described and justified?**
Yes. 5 ML baselines (XGBoost, LightGBM, RF, MLP, Logistic Regression), 2 tool baselines (Drishti heuristic rules, majority class). Drishti and WisIO are the closest existing tools. AIIO is regression (different task). IOAgent/ION are LLM-only (our ML component addresses their limitation). All baselines evaluated on the same 436 held-out benchmark test set.

**Q: Is the experimental setup reproducible?**
Yes. Dataset has public DOI (10.5281/zenodo.15052603). Code in anonymous GitHub repository. Conda environment specified in environment.yml. All LLM outputs cached as JSON (reproducible without API keys). Benchmark configurations documented in sweep scripts. Random seeds fixed and documented. Docker/conda environment for exact replication.

---

## 2. Performance Evaluation

**Q: Are appropriate metrics used and justified?**
Yes. Multi-label classification: Micro-F1 (overall, weights by frequency), Macro-F1 (equity across labels), Hamming loss, per-label F1/Precision/Recall, bootstrap 95% CI. LLM component: groundedness score (fraction of KB-verified citations), latency, token cost. Closed-loop: geometric mean speedup, per-workload speedup table. Metrics follow Hoefler & Belli (SC'15) scientific benchmarking guidelines.

**Q: Are results compared against meaningful baselines?**
Yes. ML detection: 5 baselines including Drishti (the standard HPC I/O tool). LLM recommendations: compared against raw LLM (no ML context), ML-only (no LLM), and ablation variants. Closed-loop speedup compared to AIIO (1.8x-146x) and STELLAR (near-optimal in 5 attempts).

**Q: Are performance claims supported by data?**
Yes. All claims with specific numbers: ML Micro-F1=0.920+/-0.004 (5 seeds, 436 test samples, bootstrap 95% CI [0.901, 0.943]). LLM groundedness=1.000 (Claude, Llama) on 12 workloads. Closed-loop speedup 7x-39x on IOR benchmarks. All results in paper tables with supporting figures.

---

## 3. Statistical Validity

**Q: Are error bars, confidence intervals, or significance tests reported?**
Yes. ML: 5-seed mean+/-std for all models. Bootstrap 95% CI on held-out test set (10,000 resamples). LLM: 5-run variance at temperature=0. Closed-loop: per-workload measurements.

**Q: Is variability across runs characterized?**
Yes. XGBoost: Micro-F1=0.920+/-0.004 across 5 seeds. LightGBM: 0.901+/-0.006. RF: 0.897+/-0.004. MLP: reported separately (single run for computational cost). LLM: deterministic at temperature=0 for Claude/Llama; GPT-4o std=0.276 on groundedness.

**Q: Are claims of improvement statistically significant?**
Yes. Phase 2 vs Phase 1: 0.923 vs 0.385 (bootstrap CI non-overlapping). ML vs Drishti: 0.923 vs 0.384 (2.4x). All improvements exceed 95% CI bounds.

---

## 4. Limitations and Impact

**Q: Are limitations clearly stated?**
Yes. Discussed in Threats to Validity section:
- Internal: heuristic label noise mitigated by Cleanlab (only 28/91K noisy samples)
- External: trained on Polaris (Lustre), generalization to other FS needs retraining
- Construct: Darshan provides aggregate counters, not per-operation traces
- Label alignment: S01 removed from interface_choice (documented rationale)
- LLM: GPT-4o shows 8.3% lower groundedness than Claude/Llama

**Q: Are threats to validity discussed?**
Yes. Four categories addressed: internal validity (data leakage prevented by temporal split, label noise quantified), external validity (two HPC systems: Polaris production + Delta benchmarks), construct validity (Darshan counter limitations), conclusion validity (5-seed variance, bootstrap CI).

**Q: Is societal impact considered?**
The system reduces wasted compute time on HPC facilities, improving energy efficiency and reducing carbon footprint. No negative societal impact identified. The system assists (not replaces) HPC users in optimizing I/O.

---

## 5. Reproducibility

**Q: Is source code available?**
Yes. Anonymous GitHub repository via anonymous.4open.science. Will be made public with DOI after acceptance.

**Q: Is training data available?**
Yes. Production dataset: DOI 10.5281/zenodo.15052603 (1.37M Darshan logs). Benchmark ground-truth: included in artifact with benchmark sweep scripts for reproduction.

**Q: Are software dependencies documented?**
Yes. requirements.txt (pinned versions), environment.yml (conda), INSTALL.md (step-by-step). Key versions: Python 3.9, XGBoost 2.1.4, LightGBM 4.5.0, scikit-learn 1.6.1, SHAP 0.49.1.

**Q: Is hardware documented?**
Yes. Training: NCSA Delta, AMD EPYC 7763, 128 cores, 256GB RAM, CPU only. Production data: ALCF Polaris, HPE Apollo Gen10+, A100 GPUs. Benchmarks: Delta, Lustre filesystem (12 OSTs).

---

## 6. Artifact Description (AD Appendix)

**Contribution-to-artifact mapping:**
- C1 (ML multi-label detection): trained models + training code + configs
- C2 (benchmark Knowledge Base): 623 JSON entries + construction pipeline
- C3 (SHAP-structured LLM recommendations): IOPrescriber pipeline + cached outputs
- C4 (closed-loop validation): SLURM scripts + before/after Darshan logs
- C5 (1.37M production dataset analysis): feature extraction + preprocessing pipeline

**Target badges:** Artifacts Available (Zenodo DOI) + Artifacts Evaluated Functional

---

## 7. Responsible Communication

**Q: Are results presented honestly?**
Yes. Both successes and limitations reported. MLP underperforms trees (0.840 vs 0.920). GPT-4o has lower groundedness (0.917 vs 1.000). Cleanlab found negligible noise (0.03%). Three dimensions (parallelism, file_strategy, throughput) required biquality training to achieve good F1. Closed-loop results include only completed validations.

**Q: Are comparisons fair?**
Yes. Same test set, same features, same splits, same hardware for all ML models. LLM comparison uses same prompt template, temperature=0, same workloads. Drishti run directly on test set (not reimplemented).
