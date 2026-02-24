# HPDC 2026 Paper Guide: GNN for I/O Bottleneck Analysis

---

## 1. Your Novel Contributions

| # | Contribution | Key Point |
|---|--------------|-----------|
| 1 | **Graph-Based Formulation** | First GNN for HPC I/O diagnosis; k-NN graphs from Darshan logs |
| 2 | **GNN Architecture Comparison** | GAT achieves 4.2% improvement over AIIO |
| 3 | **Multi-Method Interpretability** | Attention + GNNExplainer + Integrated Gradients consensus |
| 4 | **Scalable Construction** | O(n log n) k-NN with Ball Tree for million-scale data |

### Differentiation from AIIO

| Aspect | AIIO | Your Approach |
|--------|------|---------------|
| Job Modeling | Independent | Graph-structured |
| Dependencies | Ignored | Explicitly modeled |
| Architecture | Ensemble ML | Graph Neural Networks |
| Interpretation | SHAP (single) | Multi-method consensus |

### Positioning Statement

> "While AIIO treats I/O features independently, our GNN-based framework explicitly models job relationships through graphs, achieving 4.2% improvement in prediction accuracy with interpretable, actionable bottleneck diagnoses through multi-method consensus."

---

## 2. Papers to Cite

### Must-Cite

| Paper | Venue | Role |
|-------|-------|------|
| AIIO - Dong, Bez, Byna | HPDC 2023 | **Main baseline** |
| Gauge - Snyder et al. | SC 2020 | Clustering I/O diagnosis |
| Drishti - Bez et al. | CLUSTER 2020 | Rule-based diagnosis |
| HeteroGNN - Dutta et al. | HPDC 2023 | GNN for HPC (advisor's paper) |
| GNN Anomaly - Borghesi et al. | ICPE 2023 | GNN in HPC |

### Architecture & Methods

| Paper | Role |
|-------|------|
| Darshan - Carns et al. | Data source |
| GNNExplainer - Ying et al. | Interpretation method |
| Integrated Gradients - Sundararajan et al. | Interpretation method |
| GCN - Kipf & Welling | Architecture |
| GraphSAGE - Hamilton et al. | Architecture |
| GAT - Veličković et al. | Architecture |

---

## 3. Experiments To Do

### Critical (Must Have)

| Experiment | What to Do |
|------------|------------|
| **Temporal Validation** | Train on months 1-30, test on 31-40 (not random split) |
| **Simple k-NN Baseline** | k-NN regression without GNN to isolate GNN contribution |
| **Statistical Rigor** | 5+ runs, std dev, paired t-test, confidence intervals |

### High Priority

| Experiment | What to Do |
|------------|------------|
| **Ablation: k Value** | Test k=3,5,7,10,15 in k-NN graph construction |
| **Ablation: Distance Metric** | Compare Euclidean vs. cosine vs. Manhattan |
| **Per-Application Breakdown** | Show results by application category |
| **Scalability** | Training time, inference latency, memory vs. dataset size |

### Already Done (From Thesis)

- [x] GCN, GraphSAGE, GAT vs. AIIO comparison
- [x] Multi-method interpretability validation
- [x] IOR benchmark (6 patterns)
- [x] IO500 (4 configurations)
- [x] E2E climate model (3.4x speedup)

---

## 4. Paper Structure

```
Abstract (~200 words)

1. Introduction
   - Motivation: I/O bottleneck problem in HPC
   - AIIO limitations: treats jobs independently
   - Your insight: job similarity graphs enable better learning
   - Contributions (4 bullet points with numbers)
   - Limitations acknowledgment

2. Background & Related Work
   - Darshan profiling
   - AIIO and other ML approaches
   - GNN architectures
   - Gap: no graph-based I/O diagnosis

3. Methodology
   - k-NN graph construction from Darshan features
   - GNN architectures (GCN, GraphSAGE, GAT)
   - Multi-method interpretability framework

4. Experimental Setup
   - NERSC Cori dataset (1M jobs, 40 months)
   - Baselines: AIIO, simple k-NN
   - Metrics: RMSE, MAE, R², MAPE

5. Evaluation
   - Prediction accuracy (GAT best, 4.2% over AIIO)
   - Temporal generalization results
   - Interpretability validation
   - Case studies: IOR, IO500, E2E
   - Ablations: k value, distance metric

6. Discussion
   - Why graph structure helps
   - Practical deployment considerations
   - Limitations

7. Conclusion
```

---

## 5. Strengthening Options

### Option A: Actionable Recommendations (Recommended)

Extend diagnosis to optimization:
1. Diagnose bottleneck → 2. Map to fix → 3. Apply → 4. Measure improvement

### Option B: Online/Real-Time

- Incremental graph updates
- Streaming prediction
- Job scheduler integration

### Option C: Cross-System Transfer

- Train on Cori, test on Perlmutter
- Transfer learning with limited data

---

## 6. What to Emphasize

- **Why GNN?** Job similarity matters; graphs capture it
- **Why multi-method?** More robust than SHAP alone
- **Practical value:** Works with existing Darshan infrastructure
- **Scale:** Million jobs, O(n log n) construction

---

## 7. What to Avoid

- Don't over-claim ("best ever")
- Don't skip AIIO comparison
- Don't ignore limitations
- Don't use random train/test split only

---

## 8. Action Checklist

### Experiments
- [ ] Run temporal train/test split (months 1-30 → 31-40)
- [ ] Implement simple k-NN baseline (no GNN)
- [ ] Run 5+ trials for all methods
- [ ] Test k=3,5,7,10,15 for graph construction
- [ ] Compare distance metrics
- [ ] Break down results by application type
- [ ] Measure training time and memory scaling

### Writing
- [ ] Frame AIIO limitations clearly
- [ ] State 4 contributions with numbers
- [ ] Include ablation results
- [ ] Acknowledge limitations upfront
- [ ] Cite all must-cite papers

### Validation
- [ ] Verify 4.2% improvement holds with temporal split
- [ ] Confirm GNN beats simple k-NN baseline
- [ ] Report confidence intervals on key results
