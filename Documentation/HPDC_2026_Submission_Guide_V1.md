# HPDC 2026 Paper Submission Guide

## Research Context
**Topic:** Graph Neural Networks for I/O Bottleneck Detection in HPC Systems using Darshan Logs

---

## 1. Key Dates (HPDC 2026)

| Milestone | Date |
|-----------|------|
| Abstract Registration | **January 29, 2026** (11:59 PM AoE) |
| Paper Submission | **February 5, 2026** (11:59 PM AoE) - **FIRM** |
| Author Notification | March 31, 2026 |
| Camera-Ready Due | April 22, 2026 |
| Conference | July 13-16, 2026 (Cleveland, OH, USA) |

**Important:** The submission deadline is firm with no extensions.

---

## 2. Formatting Requirements

### Template
- **Format:** ACM sigconf proceedings template
- **LaTeX class:** `\documentclass[sigconf]{acmart}`
- **Download:** [ACM Primary Article Template](https://www.acm.org/publications/proceedings-template)

### Page Limits
- **Maximum:** 11 pages (excluding references)
- **References:** Unlimited additional pages
- **Format:** Two-column, 10pt font

### Submission Process
- Submit via HotCRP (link TBD on conference website)
- Double-blind review process
- Remove all author-identifying information from the paper
- Anonymize self-citations (use third person: "Prior work [X] showed...")

---

## 3. Content Expectations for HPDC

HPDC values papers that demonstrate:

### Balance of Theory and Practice
- **Strong systems contribution** with real-world validation
- Theoretical foundations must be backed by practical impact
- Emphasis on reproducibility and experimental rigor

### Experimental Methodology Expectations
- Large-scale evaluation (your 1M job dataset is excellent)
- Multiple baselines for comparison
- Statistical significance testing
- Real workload validation (not just synthetic benchmarks)
- Performance at scale demonstrations

### Topics of Interest (Relevant to Your Work)
- Performance modeling and prediction
- I/O and storage systems
- Machine learning for systems
- Workload characterization
- Scalable data analytics

---

## 4. Recommended Paper Structure (11 pages)

| Section | Pages | Content |
|---------|-------|---------|
| **Abstract** | 0.25 | Problem, approach, key results (GAT: RMSE 0.2521, R² 0.9342) |
| **Introduction** | 1.25 | I/O bottleneck challenge, GNN motivation, contributions |
| **Background & Related Work** | 1.5 | Darshan, GNN architectures, AIIO comparison |
| **System Design** | 2.0 | k-NN graph construction, GNN architectures, interpretability framework |
| **Experimental Setup** | 1.0 | NERSC Cori dataset, baselines, metrics |
| **Evaluation** | 3.0 | Prediction accuracy, interpretability analysis, case studies |
| **Discussion** | 1.0 | Insights, limitations, broader applicability |
| **Conclusion** | 0.5 | Summary, future directions |
| **References** | +∞ | Unlimited |

---

## 5. Experiment Design Recommendations

### Must-Have Experiments (from thesis)
1. **Prediction Accuracy Comparison**
   - GCN, GraphSAGE, GAT vs. AIIO baseline
   - Metrics: RMSE, MAE, R², MAPE
   - Your result: GAT achieves 4.2% improvement over AIIO

2. **Interpretability Validation**
   - Multi-method framework (Attention + GNNExplainer + Integrated Gradients)
   - Consensus-based bottleneck identification
   - Expert validation of identified bottlenecks

3. **Real Workload Validation**
   - IOR benchmark (6 transfer patterns)
   - IO500 (4 configurations)
   - E2E climate model (3.4x speedup demonstration)

### Recommended Additions for HPDC

4. **Ablation Study on Graph Construction**
   - Vary k in k-NN (k=3,5,7,10,15)
   - Compare distance metrics (Euclidean vs. cosine vs. Manhattan)
   - Justify k=5 choice with empirical evidence

5. **Scalability Analysis**
   - Training time vs. dataset size
   - Inference latency for online diagnosis
   - Memory footprint comparison

6. **Temporal Generalization** (Critical Gap)
   - Train on months 1-30, test on months 31-40
   - Evaluate model drift over time
   - Compare with re-training strategies

7. **Simple Baseline Comparison**
   - Add k-NN regression (without GNN) as baseline
   - Isolates GNN contribution from graph structure contribution
   - Strengthens claims about GNN necessity

8. **Per-Application Breakdown**
   - Show performance across application categories
   - Identify where GNN excels vs. struggles
   - Provides actionable insights for practitioners

---

## 6. Novel Contributions to Emphasize

### Primary Contributions
1. **First GNN-based I/O bottleneck diagnosis system**
   - Novel application of graph learning to HPC I/O analysis
   - k-NN graph construction from Darshan feature space

2. **Multi-method interpretability framework**
   - Consensus across 3 techniques (unique contribution)
   - Actionable bottleneck identification
   - Validated on real HPC workloads

3. **Large-scale empirical validation**
   - 1M jobs from production supercomputer (NERSC Cori)
   - Statistically rigorous sampling (K-S test validation)
   - 40-month longitudinal dataset

4. **End-to-end system integration**
   - From Darshan logs to optimization recommendations
   - Demonstrated 3.4x speedup on E2E climate model
   - Practical deployment considerations

### Differentiators from AIIO
| Aspect | AIIO | Your Approach |
|--------|------|---------------|
| Architecture | Traditional ML ensemble | Graph Neural Networks |
| Feature Modeling | Independent features | Relational patterns via graphs |
| Interpretability | Single method | Multi-method consensus |
| Validation Scale | Limited workloads | 1M jobs + real applications |

---

## 7. Gaps to Address

### Critical Gaps
1. **Temporal Validation**
   - Current: Random 80/20 split
   - Needed: Time-based train/test split
   - Why: Real deployment requires predicting future workloads

2. **Simple Baseline Missing**
   - Add: k-NN regression without GNN
   - Why: Proves GNN learns beyond neighborhood averaging

3. **Per-Application Analysis**
   - Add: Breakdown by application type/domain
   - Why: Shows where approach works best

### Nice-to-Have Extensions
4. **Online Learning**
   - Incremental model updates
   - Adaptation to workload drift

5. **Multi-System Generalization**
   - Test on other supercomputers (if data available)
   - Cross-system transfer learning

6. **Resource Overhead Analysis**
   - Training cost vs. prediction benefit
   - Break-even analysis for deployment

---

## 8. Positioning Against Related Work

### Key Papers to Cite and Differentiate

| Work | Their Approach | Your Differentiation |
|------|----------------|---------------------|
| AIIO (SC'20) | ML ensemble for I/O prediction | GNN captures relational patterns; multi-method interpretability |
| Umami | Feature extraction from Darshan | You build graphs from features; enable GNN learning |
| TOKIO | I/O analysis framework | Your focus on prediction + interpretation |
| General GNN papers | Various domains | First application to HPC I/O diagnosis |

### Positioning Statement
> "While existing approaches like AIIO treat I/O features independently, our GNN-based framework explicitly models relationships between job characteristics, achieving 4.2% improvement in prediction accuracy while providing interpretable, actionable bottleneck diagnoses through a novel multi-method consensus framework."

---

## 9. Submission Checklist

### Before Abstract Registration (Jan 29)
- [ ] Finalize paper title
- [ ] Write compelling 200-word abstract
- [ ] Identify all co-authors and affiliations
- [ ] Register on HotCRP

### Before Submission (Feb 5)
- [ ] Complete all experiments including:
  - [ ] Temporal validation (train/test split by time)
  - [ ] Simple k-NN baseline comparison
  - [ ] Per-application performance breakdown
- [ ] Paper formatted in ACM sigconf template
- [ ] 11 pages or fewer (excluding references)
- [ ] All figures are readable in print (not just color)
- [ ] Double-blind compliance:
  - [ ] No author names/affiliations in paper
  - [ ] Self-citations in third person
  - [ ] No identifying acknowledgments
- [ ] Code/data anonymized for reproducibility section
- [ ] PDF validated for submission system

### Quality Checks
- [ ] Clear problem statement and motivation
- [ ] Well-defined contributions (3-4 bullet points)
- [ ] Comprehensive related work comparison
- [ ] Statistical significance of results
- [ ] Limitations honestly discussed
- [ ] Reproducibility information provided

---

## 10. Additional Resources

- **HPDC 2026 Website:** https://www.hpdc.org/2026/
- **ACM Template:** https://www.acm.org/publications/proceedings-template
- **Darshan Documentation:** https://www.mcs.anl.gov/research/projects/darshan/
- **Previous HPDC Proceedings:** https://dl.acm.org/conference/hpdc

---

*Generated: January 2026*
*Target Venue: 35th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC 2026)*
