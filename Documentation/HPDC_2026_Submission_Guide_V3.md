# Comprehensive Guide: HPDC 2026 Paper Submission
## GNN-Based I/O Bottleneck Analysis for HPC Systems

---

## Table of Contents
1. [Key Dates](#1-key-dates)
2. [Conference Overview](#2-conference-overview)
3. [Formatting Requirements](#3-formatting-requirements)
4. [Paper Structure](#4-paper-structure)
5. [What HPDC Reviewers Expect](#5-what-hpdc-reviewers-expect)
6. [Your Positioning and Novel Contributions](#6-your-positioning-and-novel-contributions)
7. [Related Papers to Reference](#7-related-papers-to-reference)
8. [Experiment Design Recommendations](#8-experiment-design-recommendations)
9. [Gaps to Address and Strengthening Options](#9-gaps-to-address-and-strengthening-options)
10. [Suggested Paper Titles](#10-suggested-paper-titles)
11. [Submission Checklist](#11-submission-checklist)
12. [Additional Resources](#12-additional-resources)

---

## 1. Key Dates

| Milestone | Date |
|-----------|------|
| Abstract Registration | **January 29, 2026** (11:59 PM AoE) |
| Paper Submission | **February 5, 2026** (11:59 PM AoE) - **FIRM** |
| Author Notification | March 31, 2026 |
| Camera-Ready Due | April 30, 2026 |
| Conference | July 13-16, 2026 (Cleveland, OH, USA) |

> **Warning:** The submission deadline is FIRM - no extensions will be granted.

---

## 2. Conference Overview

### Basic Information
- **Conference:** 35th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC 2026)
- **Location:** Cleveland, OH, USA (Case Western Reserve University)
- **Submission Categories:**
  1. **Regular Papers** - New research ideas with experimental implementation/evaluation
  2. **Open-source Tools and Data Papers** - Design, development, and evaluation of open-source tools/datasets

### Topics of Interest (Relevant to Your Work)
- File and storage systems, I/O, and data management
- Performance modeling and benchmarking
- Scientific applications, algorithms, and workflows
- Parallel and distributed computing systems and software for AI

> **Important Note from CFP:** "HPDC welcomes submissions that utilize Artificial Intelligence to enhance the topics of interest mentioned above... However, we emphasize that an HPDC submission must clearly address and explain its connection to research in parallel and distributed computing."

### Review Process
- **Dual-anonymous (double-blind) reviewing** - Authors and reviewers identities are hidden
- You must anonymize your paper completely
- Use paper ID instead of author names
- Refer to your own work in third person

---

## 3. Formatting Requirements

### Page Limits
| Type | Limit |
|------|-------|
| Main content | **11 pages maximum** (excluding references) |
| References | Unlimited additional pages |
| Camera-ready bonus | +1 page for addressing reviewer feedback |
| AD/AE Appendix | Optional +2 pages for Artifact Description/Evaluation |

### LaTeX Template

```latex
% FOR SUBMISSION (REQUIRED - single column review format)
\documentclass[manuscript,review,anonymous]{acmart}
\acmSubmissionID{paper-ID}

% AFTER ACCEPTANCE (two-column format)
\documentclass[sigconf]{acmart}
```

**Download:** https://www.acm.org/publications/proceedings-template
**Overleaf:** https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc

### Critical Formatting Rules
- Do NOT modify margins, typeface sizes, or line spacing
- Do NOT define your own LaTeX commands with `\newcommand`
- All figures MUST have `\Description{}` commands for accessibility
- All figures must be readable in print (not just color)
- Paper type declaration required in title:
```latex
\title{Your Paper Title\\
Paper Type: Regular}
```

### Submission Process
- Submit via HotCRP (link on conference website)
- Double-blind review process
- Remove all author-identifying information
- Anonymize self-citations (use third person: "Prior work [X] showed...")

---

## 4. Paper Structure

### 4.1 Page Allocation Overview

| Section | Pages | Key Content |
|---------|-------|-------------|
| **Abstract** | ~0.25 | Problem, approach, key results (GAT: RMSE 0.2521, R² 0.9342) |
| **Introduction** | 2-2.5 | Follow HPDC-recommended structure below |
| **Background & Related Work** | 1.5-2 | Darshan, AIIO, GNN architectures, research gap |
| **System Design** | 2-2.5 | Graph construction, GNN architecture, interpretability |
| **Experimental Setup** | 1-1.5 | Dataset, hardware, baselines, metrics |
| **Evaluation** | 2.5-3 | Accuracy, diagnosis validation, case studies, ablations |
| **Discussion** | 0.5-1 | Key findings, implications, limitations |
| **Conclusion** | 0.5 | Summary, future directions |
| **References** | ∞ | Unlimited |

### 4.2 HPDC-Recommended Introduction Structure

HPDC 2026 strongly recommends this introduction structure:

```
1. MOTIVATION
   - Clearly state the objective
   - Provide quantitative support for the specific problem

2. LIMITATIONS OF STATE-OF-THE-ART
   - Review most relevant recent prior works (especially AIIO)
   - Articulate limitations of previous work
   - Explain how your approach breaks these limitations

3. KEY INSIGHTS AND CONTRIBUTIONS
   - Major insights enabling your approach
   - Novelty of these insights
   - Key ideas of your approach
   - List contributions with flagship empirical results
   - Improvement percentages over prior art

4. EXPERIMENTAL METHODOLOGY AND ARTIFACT AVAILABILITY
   - Key experimental infrastructure
   - Methodology choices with citations
   - Statement about artifact availability

5. LIMITATIONS OF THE PROPOSED APPROACH
   - Acknowledge limitations and scope for improvement
   - Identify conclusions sensitive to assumptions
```

### 4.3 Full Paper Outline

```
Abstract (~200 words)

1. Introduction (2-2.5 pages)
   1.1 Motivation
   1.2 Limitations of Existing Approaches
   1.3 Key Insights and Contributions
   1.4 Experimental Overview
   1.5 Limitations

2. Background and Related Work (1.5-2 pages)
   2.1 HPC I/O Profiling (Darshan)
   2.2 Machine Learning for I/O Analysis (AIIO, Gauge)
   2.3 Graph Neural Networks
   2.4 Research Gap

3. System Design / Methodology (2-2.5 pages)
   3.1 Problem Formulation
   3.2 Graph Construction from Darshan Logs
   3.3 GNN Architecture (GCN, GraphSAGE, GAT)
   3.4 Multi-Method Interpretability Framework

4. Experimental Setup (1-1.5 pages)
   4.1 Workloads and Datasets
   4.2 Hardware and System Environment
   4.3 Software Stack
   4.4 Baselines
   4.5 Methodology and Metrics

5. Evaluation Results (2.5-3 pages)
   5.1 Performance Prediction Accuracy
   5.2 Bottleneck Diagnosis Validation
   5.3 Case Studies (IOR, IO500, E2E Climate Model)
   5.4 Ablation Studies

6. Discussion (0.5-1 page)
   6.1 Key Findings
   6.2 Implications for HPC Practitioners
   6.3 Limitations

7. Conclusion (0.5 page)

References (unlimited)

[Optional] Artifact Description / Artifact Evaluation Appendix (up to 2 pages)
```

---

## 5. What HPDC Reviewers Expect

### Evaluation Criteria
Your paper will be evaluated on:

| Criterion | What Reviewers Look For |
|-----------|------------------------|
| **Novelty** | What is genuinely new about your approach? |
| **Scientific Value** | Is the methodology sound? |
| **Demonstrated Usefulness** | Does it work in practice? |
| **Potential Impact** | Will this matter to the HPC community? |

### What Makes Papers Get Rejected

| Issue | Description |
|-------|-------------|
| Incremental | Improvement without clear novelty |
| Weak evaluation | Insufficient experimental evidence |
| Missing baselines | No comparison to state-of-the-art (especially AIIO) |
| Poor HPC connection | Weak link to parallel/distributed computing |
| Over-claiming | Claims without supporting evidence |
| No ablations | Missing component analysis |

### What Makes Papers Get Accepted

| Strength | Description |
|----------|-------------|
| Clear novelty | Novel insight articulated (not just better numbers) |
| Strong experiments | Real systems, multiple baselines |
| Fair comparison | Recent baselines, same conditions |
| Practical validation | Real applications, not just synthetic |
| Reproducibility | Artifact availability statement |
| HPC relevance | Clear connection to I/O systems |

---

## 6. Your Positioning and Novel Contributions

### Your Main Baseline: AIIO (HPDC 2023)

**Full Citation:**
> Bin Dong, Jean Luca Bez, and Suren Byna. 2023. "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis." In Proceedings of the 32nd International Symposium on High-Performance Parallel and Distributed Computing (HPDC '23).

**AIIO's Approach:**
- Uses ensemble ML (XGBoost, LightGBM, CatBoost, MLP, TabNet)
- Uses SHAP for interpretation
- **Key Limitation:** Treats each job independently - ignores structural dependencies

### Your Novel Contributions

| # | Contribution | Details |
|---|--------------|---------|
| 1 | **Graph-Based Formulation** (Novel) | First to use GNN for HPC I/O bottleneck diagnosis; k-NN similarity graphs from Darshan logs; explicitly models inter-job dependencies |
| 2 | **GNN Architecture Comparison** | Systematic evaluation of GCN, GraphSAGE, GAT; shows attention mechanisms are essential; GAT achieves 4.2% improvement over AIIO |
| 3 | **Multi-Method Interpretability** (Novel) | Combines Attention + GNNExplainer + Integrated Gradients; consensus-based bottleneck identification; more robust than single-method (SHAP alone) |
| 4 | **Scalable Graph Construction** | O(n log n) k-NN with Ball Tree; handles million-scale datasets |

### Differentiation from AIIO

| Aspect | AIIO | Your Approach |
|--------|------|---------------|
| Job Modeling | Independent | Graph-structured |
| Dependencies | Ignored | Explicitly modeled |
| Architecture | Ensemble ML | Graph Neural Networks |
| Feature Modeling | Independent features | Relational patterns via graphs |
| Interpretation | SHAP (single method) | Multi-method consensus |
| Validation Scale | Limited workloads | 1M jobs + real applications |

### Key Insight (Positioning Statement)

> "Jobs with similar I/O patterns share similar bottlenecks. By explicitly modeling these relationships through graphs, GNNs can learn from neighborhood information, improving both prediction and diagnosis. While existing approaches like AIIO treat I/O features independently, our GNN-based framework explicitly models relationships between job characteristics, achieving 4.2% improvement in prediction accuracy while providing interpretable, actionable bottleneck diagnoses through a novel multi-method consensus framework."

---

## 7. Related Papers to Reference

### Must-Cite Papers

| # | Paper | Venue | Relevance |
|---|-------|-------|-----------|
| 1 | **AIIO** - Dong, Bez, Byna. "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis" | HPDC 2023 | **Your main baseline** |
| 2 | **Gauge** - Snyder et al. "HPC I/O throughput bottleneck analysis with explainable local models" | SC 2020 | Clustering approach to I/O diagnosis |
| 3 | **Drishti** - Bez et al. "Drishti: Guiding End-Users in the I/O Optimization Journey" | CLUSTER 2020 | Rule-based I/O diagnosis |
| 4 | **HeteroGNN** - Dutta, Alcaraz, TehraniJamsaz, César, Sikora, Jannesari. "Performance Optimization using Multimodal Modeling and Heterogeneous GNN" | HPDC 2023 | Your advisor's paper - GNN for HPC |
| 5 | **GNN Anomaly** - Borghesi et al. "Graph Neural Networks for Anomaly Anticipation in HPC Systems" | ICPE 2023 | GNN in HPC context |

### Other Essential References

| Paper | Why Cite |
|-------|----------|
| **Darshan** - Carns et al. | Foundation for your data |
| **GNNExplainer** - Ying et al. | Your interpretation method |
| **Integrated Gradients** - Sundararajan et al. | Your interpretation method |
| **GCN** - Kipf & Welling | Architecture reference |
| **GraphSAGE** - Hamilton et al. | Architecture reference |
| **GAT** - Veličković et al. | Architecture reference |
| **Umami** | Feature extraction from Darshan |
| **TOKIO** | I/O analysis framework |

---

## 8. Experiment Design Recommendations

### 8.1 What You Already Have (From Thesis)

| Component | Details |
|-----------|---------|
| Dataset | 1M Darshan logs from NERSC Cori (40 months) |
| Models | GAT, GraphSAGE, GCN implementation |
| Benchmarks | IOR (6 patterns), IO500 (4 configs) |
| Case Study | E2E climate model (3.4x speedup) |
| Metrics | RMSE 0.2521, R² 0.9342, 4.2% improvement over AIIO |

### 8.2 Must-Have Experiments

| Experiment | Purpose | Your Status |
|------------|---------|-------------|
| Prediction Accuracy | GCN, GraphSAGE, GAT vs. AIIO | ✅ Done |
| Interpretability Validation | Multi-method consensus | ✅ Done |
| Real Workload Validation | IOR, IO500, E2E | ✅ Done |

### 8.3 Recommended Additions for HPDC

| # | Experiment | Details | Priority |
|---|------------|---------|----------|
| 1 | **Ablation: Graph Construction** | Vary k in k-NN (k=3,5,7,10,15); compare distance metrics (Euclidean vs. cosine vs. Manhattan) | High |
| 2 | **Scalability Analysis** | Training time vs. dataset size; inference latency; memory footprint | High |
| 3 | **Temporal Generalization** | Train on months 1-30, test on months 31-40; evaluate model drift | **Critical** |
| 4 | **Simple Baseline** | k-NN regression without GNN (isolates GNN contribution) | **Critical** |
| 5 | **Per-Application Breakdown** | Performance across application categories | High |
| 6 | **Statistical Rigor** | 5+ runs with std dev; paired t-test/Wilcoxon; confidence intervals | **Critical** |
| 7 | **Additional Benchmarks** | Consider: FLASH-IO, VPIC-IO, other DOE applications | Medium |

### 8.4 Experimental Setup Section Requirements

Per HPDC CFP, your experimental setup must include:

**Workloads and Datasets:**
- Dataset source, size, time period
- Preprocessing steps (K-S test validation)
- Train/val/test splits
- Why representative of HPC use cases

**Hardware and System Environment:**
- GPU specifications (NVIDIA H200)
- CPU, memory configuration
- Storage system details
- Node count, topology

**Software Stack:**
- PyTorch/PyTorch Geometric versions
- Darshan version used
- All library versions
- Build commands (or link to artifact)

**Baselines:**
- Exact same preprocessing for all methods
- Tuned hyperparameters documented
- Validation that reproduction matches published results

**Methodology:**
- Primary metrics (RMSE, MAE, R², MAPE)
- Number of trials (minimum 5)
- Warm-up policy
- Outlier handling

---

## 9. Gaps to Address and Strengthening Options

### 9.1 Critical Gaps to Address

| Gap | Current State | What to Add | Why Important |
|-----|---------------|-------------|---------------|
| **Temporal Validation** | Random 80/20 split | Time-based train/test split | Real deployment requires predicting future workloads |
| **Simple Baseline** | Missing | k-NN regression without GNN | Proves GNN learns beyond neighborhood averaging |
| **Per-Application Analysis** | Aggregate results | Breakdown by application type | Shows where approach works best |
| **Statistical Significance** | Limited | 5+ runs, t-tests, confidence intervals | Required for HPDC acceptance |

### 9.2 Nice-to-Have Extensions

| Extension | Details |
|-----------|---------|
| Online Learning | Incremental model updates, adaptation to workload drift |
| Multi-System Generalization | Test on other supercomputers, cross-system transfer learning |
| Resource Overhead Analysis | Training cost vs. prediction benefit, break-even analysis |

### 9.3 Options for Strengthening Your Contribution

#### Option A: Enhanced Interpretability (Recommended)

**New Contribution:** Automated I/O Optimization Recommendations

Extend beyond diagnosis to automatic recommendations:
1. Diagnose bottleneck (what you have)
2. Map to optimization strategies
3. Validate by implementing recommendations
4. Show measured improvement

> **Positioning:** "Not just better prediction, but actionable diagnosis that leads to measured performance improvements"

#### Option B: Real-Time / Online Diagnosis

Adapt your approach for online use:
- Incremental graph updates
- Streaming prediction
- Lower latency diagnosis
- Integration with job scheduler

#### Option C: Cross-System Generalization

Show your model generalizes:
- Train on Cori, test on Perlmutter (or another system)
- Transfer learning experiments
- Fine-tuning with limited data

#### Option D: Larger Scale Evaluation

Use the full 6.6M dataset:
- Distributed GNN training
- Show scaling to production scale
- Compare graph construction strategies

### Recommendation

For maximum acceptance probability:
1. **Keep core contribution** (GNN for I/O diagnosis with multi-method interpretability)
2. **Address critical gaps** (temporal validation, simple baseline, statistical rigor)
3. **Add Option A** if time permits (actionable recommendations with measured improvement)

---

## 10. Suggested Paper Titles

Based on your work, consider:

1. **"Graph Neural Networks for Interpretable I/O Bottleneck Diagnosis in High-Performance Computing"**

2. **"GNN-IO: Leveraging Job Similarity Graphs for Automated I/O Performance Diagnosis"**

3. **"Beyond Independent Jobs: Graph-Based Machine Learning for HPC I/O Bottleneck Analysis"**

4. **"Attention-Based Graph Neural Networks for Scalable I/O Performance Diagnosis in HPC Systems"**

---

## 11. Submission Checklist

### Phase 1: Before Writing

- [ ] Read AIIO paper thoroughly (your main baseline)
- [ ] Read 3-5 recent HPDC papers for style/depth expectations
- [ ] Identify 3-4 clear novel contributions
- [ ] Plan experiments to support each claim

### Phase 2: Before Abstract Registration (January 29, 2026)

- [ ] Finalize paper title (include "Paper Type: Regular")
- [ ] Write compelling ~200-word abstract
- [ ] Identify all co-authors and affiliations
- [ ] All co-authors have ORCID IDs
- [ ] Register on HotCRP

### Phase 3: Experiments (Before February 5)

- [ ] Complete all must-have experiments
- [ ] **Temporal validation** (train/test split by time)
- [ ] **Simple k-NN baseline** comparison
- [ ] **Per-application performance** breakdown
- [ ] **Statistical rigor**: 5+ runs, std dev, significance tests
- [ ] Ablation study on graph construction (k values, distance metrics)
- [ ] Scalability analysis (time, memory)

### Phase 4: Paper Content

- [ ] Abstract: Problem, approach, key results, impact
- [ ] Introduction follows HPDC-recommended 5-part structure
- [ ] Clear statement of novelty vs. AIIO
- [ ] Graph construction methodology detailed
- [ ] GNN architecture clearly explained
- [ ] Interpretability framework described
- [ ] All baselines reproduced and compared fairly
- [ ] Statistical significance reported with confidence intervals
- [ ] Ablation study included
- [ ] Real application case studies
- [ ] Limitations acknowledged honestly
- [ ] Connection to parallel/distributed computing explicit

### Phase 5: Formatting

- [ ] Uses ACM template: `\documentclass[manuscript,review,anonymous]{acmart}`
- [ ] `\acmSubmissionID{paper-ID}` included
- [ ] Maximum 11 pages (excluding references)
- [ ] All figures have `\Description{}` for accessibility
- [ ] All figures readable in print (not just color)
- [ ] High-quality figures (vector when possible)
- [ ] No custom `\newcommand` definitions

### Phase 6: Anonymization

- [ ] No author names or affiliations in paper
- [ ] No acknowledgments section
- [ ] Own work cited in third person ("Prior work [X]...")
- [ ] No institution-identifying details
- [ ] GitHub/code links anonymized or removed
- [ ] No "our previous work" language

### Phase 7: Final Submission (February 5, 2026)

- [ ] PDF validates correctly in submission system
- [ ] Conflict of interest declared
- [ ] Artifact availability statement included
- [ ] All co-author information entered

### Quality Self-Check

- [ ] Clear problem statement and motivation
- [ ] Well-defined contributions (3-4 bullet points)
- [ ] Comprehensive related work comparison
- [ ] Results support all claims made
- [ ] Limitations honestly discussed
- [ ] Reproducibility information provided

---

## 12. Additional Resources

| Resource | Link |
|----------|------|
| HPDC 2026 Website | https://www.hpdc.org/2026/ |
| ACM Template | https://www.acm.org/publications/proceedings-template |
| Overleaf Template | https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc |
| Darshan Documentation | https://www.mcs.anl.gov/research/projects/darshan/ |
| Previous HPDC Proceedings | https://dl.acm.org/conference/hpdc |

---

## Summary: Your Path to Acceptance

### Core Selling Points
1. **Novel Insight:** Job similarity matters - graph structure captures it
2. **Technical Contribution:** First GNN approach for I/O diagnosis
3. **Robustness:** Multi-method interpretability beats single-method
4. **Practical Impact:** 3.4x improvement on real application

### What to Emphasize
- Connection to parallel/distributed computing (HPC I/O is inherently parallel)
- Practical deployment value (works with existing Darshan infrastructure)
- Interpretability (actionable for system administrators)
- Scale (million-job dataset, O(n log n) construction)

### What to Be Careful About
- Don't over-claim ("best ever") - be precise about improvements
- Acknowledge limitations upfront
- Explain why GNN is appropriate (not just that it works)
- Show clear improvement over AIIO with fair comparison

---

*Generated: January 2026*
*Target Venue: 35th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC 2026)*
*Deadline: February 5, 2026 (FIRM)*
