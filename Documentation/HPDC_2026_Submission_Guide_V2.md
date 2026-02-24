# Comprehensive Guide: HPDC 2026 Paper Submission
## GNN-Based I/O Bottleneck Analysis for HPC Systems

---

## Table of Contents
1. [Conference Requirements Overview](#1-conference-requirements-overview)
2. [Key Dates](#2-key-dates)
3. [Paper Format and Template](#3-paper-format-and-template)
4. [Required Paper Structure](#4-required-paper-structure)
5. [What HPDC Reviewers Expect](#5-what-hpdc-reviewers-expect)
6. [Your Paper: Positioning and Novelty](#6-your-paper-positioning-and-novelty)
7. [Similar/Related Papers to Reference](#7-similarrelated-papers-to-reference)
8. [Experimental Design Recommendations](#8-experimental-design-recommendations)
9. [How to Strengthen Your Contribution](#9-how-to-strengthen-your-contribution)
10. [Checklist for Submission](#10-checklist-for-submission)

---

## 1. Conference Requirements Overview

### Basic Information
- **Conference:** 35th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC 2026)
- **Location:** Cleveland, OH, USA (Case Western Reserve University)
- **Dates:** July 13-16, 2026
- **Submission Categories:**
  1. **Regular Papers** - New research ideas with experimental implementation/evaluation
  2. **Open-source Tools and Data Papers** - Design, development, and evaluation of open-source tools/datasets

### Review Process
- **Dual-anonymous (double-blind) reviewing** - Authors and reviewers identities are hidden
- You must anonymize your paper completely
- Use paper ID instead of author names
- Refer to your own work in third person

### Topics of Interest (Relevant to Your Work)
✅ **File and storage systems, I/O, and data management**  
✅ **Performance modeling and benchmarking**  
✅ **Scientific applications, algorithms, and workflows**  
✅ **Parallel and distributed computing systems and software for AI**  

> **Important Note from CFP:** "HPDC welcomes submissions that utilize Artificial Intelligence to enhance the topics of interest mentioned above... However, we emphasize that an HPDC submission must clearly address and explain its connection to research in parallel and distributed computing."

---

## 2. Key Dates

| Deadline | Date |
|----------|------|
| Abstract Registration | January 29, 2026 (AoE) |
| Paper Submission | February 5, 2026 (FIRM, AoE) |
| Notification of Acceptance | March 31, 2026 |
| Camera-ready Version | April 30, 2026 |

⚠️ **The submission deadline is FIRM** - no extensions will be granted.

---

## 3. Paper Format and Template

### Page Limits
- **Maximum 11 pages** (excluding references)
- **Additional 1 page** allowed in camera-ready for addressing reviewer feedback
- **Optional 2 pages** for Artifact Description (AD) / Artifact Evaluation (AE) appendix

### LaTeX Template
Use the ACM sigconf template:

```latex
% For submission (REQUIRED - single column review format)
\documentclass[manuscript,review,anonymous]{acmart}

% After acceptance (two-column format)
\documentclass[sigconf]{acmart}
```

**Download from:** https://www.acm.org/publications/proceedings-template

**Overleaf Template:** https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc

### Critical Formatting Rules
- Do NOT modify margins, typeface sizes, or line spacing
- Do NOT define your own LaTeX commands with `\newcommand`
- Include `\acmSubmissionID{paper-ID}` for submission
- All figures must have `\Description{}` commands for accessibility
- References section does not count toward page limit

### Paper Type Declaration
Add to your title (required):
```latex
\title{Your Paper Title\\
Paper Type: Regular}
```

---

## 4. Required Paper Structure

HPDC 2026 strongly recommends this introduction structure:

### 4.1 Introduction Section Structure

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

### 4.2 Full Paper Structure

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
   3.3 GNN Architecture
   3.4 Interpretability Framework

4. Experimental Setup (1-1.5 pages)
   4.1 Workloads and Datasets
   4.2 Hardware and System Environment
   4.3 Software Stack
   4.4 Baselines
   4.5 Methodology and Metrics

5. Evaluation Results (2.5-3 pages)
   5.1 Performance Prediction Accuracy
   5.2 Bottleneck Diagnosis Validation
   5.3 Case Studies
   5.4 Ablation Studies

6. Discussion (0.5-1 page)
   6.1 Key Findings
   6.2 Implications
   6.3 Limitations

7. Conclusion (0.5 page)

References (unlimited)
```

---

## 5. What HPDC Reviewers Expect

### Evaluation Criteria
Based on HPDC's guidelines, your paper will be evaluated on:

1. **Novelty** - What is genuinely new about your approach?
2. **Scientific Value** - Is the methodology sound?
3. **Demonstrated Usefulness** - Does it work in practice?
4. **Potential Impact** - Will this matter to the HPC community?

### What Makes Papers Get Rejected

❌ Incremental improvement without clear novelty  
❌ Weak experimental evaluation  
❌ No comparison to state-of-the-art (especially AIIO)  
❌ Poor connection to parallel/distributed computing  
❌ Over-claiming without evidence  
❌ Missing ablation studies  

### What Makes Papers Get Accepted

✅ Clear articulation of novel insight (not just better numbers)  
✅ Strong experimental methodology with real systems  
✅ Fair comparison with recent baselines  
✅ Practical validation on real applications  
✅ Reproducible results (artifact availability)  
✅ Clear connection to HPC I/O systems  

---

## 6. Your Paper: Positioning and Novelty

### Your Main Baseline: AIIO (HPDC 2023)

**Citation:**
> Bin Dong, Jean Luca Bez, and Suren Byna. 2023. AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis. In HPDC '23.

**AIIO's Approach:**
- Uses ensemble ML (XGBoost, LightGBM, CatBoost, MLP, TabNet)
- Uses SHAP for interpretation
- **Key Limitation:** Treats each job independently - ignores structural dependencies

### Your Novel Contributions (Based on Your Thesis)

1. **Graph-Based Formulation** (Novel)
   - First to use GNN for HPC I/O bottleneck diagnosis
   - Constructs k-NN similarity graphs from Darshan logs
   - Explicitly models inter-job dependencies

2. **GNN Architecture Comparison** (Contribution)
   - Systematic evaluation of GCN, GraphSAGE, GAT
   - Shows attention mechanisms are essential
   - GAT achieves 4.2% improvement over AIIO

3. **Multi-Method Interpretability** (Novel)
   - Combines attention weights + GNNExplainer + Integrated Gradients
   - Consensus-based bottleneck identification
   - More robust than single-method (SHAP alone)

4. **Scalable Graph Construction** (Methodological)
   - O(n log n) k-NN with Ball Tree
   - Handles million-scale datasets

### How to Position Your Novelty

**Key Insight:** Jobs with similar I/O patterns share similar bottlenecks. By explicitly modeling these relationships through graphs, GNNs can learn from neighborhood information, improving both prediction and diagnosis.

**Differentiation from AIIO:**
| Aspect | AIIO | Your Approach |
|--------|------|---------------|
| Job Modeling | Independent | Graph-structured |
| Dependencies | Ignored | Explicitly modeled |
| Interpretation | SHAP (single) | Multi-method consensus |
| Architecture | Ensemble ML | Graph Neural Networks |

---

## 7. Similar/Related Papers to Reference

### Must-Cite Papers

1. **AIIO (HPDC 2023)** - Your main baseline
   > Dong, Bez, Byna. "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis."

2. **Gauge (SC 2020)** - Clustering approach to I/O diagnosis
   > Snyder et al. "HPC I/O throughput bottleneck analysis with explainable local models."

3. **Drishti (CLUSTER 2020)** - Rule-based I/O diagnosis
   > Bez et al. "Drishti: Guiding End-Users in the I/O Optimization Journey."

4. **Performance Optimization using Multimodal Modeling and Heterogeneous GNN (HPDC 2023)**
   > Dutta, Alcaraz, TehraniJamsaz, César, Sikora, Jannesari. (Your advisor's paper!)

5. **GNN for Anomaly Anticipation in HPC (ICPE 2023)**
   > Borghesi et al. "Graph Neural Networks for Anomaly Anticipation in HPC Systems."

### Other Relevant Works

6. **Darshan** - Carns et al. (Foundation for your data)
7. **GNNExplainer** - Ying et al. (Your interpretation method)
8. **Integrated Gradients** - Sundararajan et al.
9. **GNN Architectures** - Kipf & Welling (GCN), Hamilton et al. (GraphSAGE), Veličković et al. (GAT)

---

## 8. Experimental Design Recommendations

### What You Already Have (From Thesis)
- 1M Darshan logs from NERSC Cori
- GNN implementation (GAT, GraphSAGE, GCN)
- IOR and IO500 benchmark validation
- E2E climate model case study

### What You Need to Strengthen for HPDC

#### 8.1 Stronger Baselines
Your current baselines are good, but consider adding:
- **Direct reproduction of AIIO** on your dataset
- **More recent methods** if available
- **Ablation variants** of your own approach

#### 8.2 More Comprehensive Benchmarks
```
Current: IOR, IO500, E2E
Consider Adding:
- FLASH-IO (astrophysics checkpoint)
- VPIC-IO (particle physics)
- Multiple real DOE applications
- Different I/O patterns systematically
```

#### 8.3 Statistical Rigor
Required for HPDC:
- Multiple runs (at least 5) with standard deviation
- Statistical significance tests (paired t-test, Wilcoxon)
- Confidence intervals on key results
- Clear explanation of variance

#### 8.4 Scalability Analysis
Add experiments showing:
- Performance with increasing dataset size
- Graph construction time scaling
- Training time vs. dataset size
- Memory usage analysis

#### 8.5 Generalization Study
Important to show:
- Cross-validation results
- Performance on unseen application types
- Temporal generalization (train on old, test on new)

### Experimental Setup Section Requirements

Per HPDC CFP, include:

**Workloads and Datasets:**
- Dataset source, size, time period
- Preprocessing steps
- Train/val/test splits
- Why representative of HPC use cases

**Hardware and System Environment:**
- GPU specifications (you used NVIDIA H200)
- CPU, memory configuration
- Storage system details
- Node count, topology

**Software Stack:**
- PyTorch/PyTorch Geometric versions
- Darshan version used
- All library versions
- Build commands (or link to artifact)

**Baselines:**
- Exact same preprocessing
- Tuned hyperparameters
- Validation that your reproduction matches published results

**Methodology:**
- Primary metrics (RMSE, MAE, R²)
- Number of trials
- Warm-up policy
- Outlier handling

---

## 9. How to Strengthen Your Contribution

### Option A: Enhanced Interpretability (Recommended)

**New Contribution:** Automated I/O Optimization Recommendations

Extend beyond diagnosis to automatic recommendations:
1. Diagnose bottleneck (what you have)
2. Map to optimization strategies
3. Validate by implementing recommendations
4. Show measured improvement

This makes it more practical and impactful.

### Option B: Real-Time / Online Diagnosis

Adapt your approach for online use:
- Incremental graph updates
- Streaming prediction
- Lower latency diagnosis
- Integration with job scheduler

### Option C: Cross-System Generalization

Show your model generalizes:
- Train on Cori, test on Perlmutter (or another system)
- Transfer learning experiments
- Fine-tuning with limited data

### Option D: Larger Scale Evaluation

Use the full 6.6M dataset:
- Distributed GNN training
- Show scaling to production scale
- Compare graph construction strategies

### My Recommendation

For maximum acceptance probability, I suggest:

1. **Keep core contribution** (GNN for I/O diagnosis with multi-method interpretability)
2. **Add actionable recommendations** (translate diagnosis to optimization actions)
3. **Strengthen E2E validation** (show closed-loop: diagnose → recommend → implement → measure improvement)
4. **Add more real applications** beyond E2E

This positions your paper as:
> "Not just better prediction, but actionable diagnosis that leads to measured performance improvements"

---

## 10. Checklist for Submission

### Before Writing

- [ ] Read AIIO paper thoroughly (your main baseline)
- [ ] Read 3-5 recent HPDC papers for style/depth
- [ ] Identify 3 clear novel contributions
- [ ] Plan experiments to support each claim

### Paper Content

- [ ] Title includes "Paper Type: Regular"
- [ ] Abstract: Problem, approach, key results, impact (~200 words)
- [ ] Introduction follows HPDC-recommended structure
- [ ] Clear statement of novelty vs. AIIO
- [ ] Graph construction methodology detailed
- [ ] GNN architecture clearly explained
- [ ] Interpretability framework described
- [ ] All baselines reproduced and compared fairly
- [ ] Statistical significance reported
- [ ] Ablation study included
- [ ] Real application case study
- [ ] Limitations acknowledged
- [ ] Connection to parallel/distributed computing explicit

### Formatting

- [ ] Uses ACM sigconf template
- [ ] `\documentclass[manuscript,review,anonymous]{acmart}`
- [ ] Maximum 11 pages (excluding references)
- [ ] All figures have `\Description{}`
- [ ] Paper ID used instead of author names
- [ ] No identifying information in text
- [ ] References to own work in third person
- [ ] High-quality figures (vector when possible)

### Anonymization

- [ ] No author names or affiliations
- [ ] No acknowledgments section
- [ ] Own work cited in third person
- [ ] No institution-identifying details
- [ ] GitHub/code links anonymized or removed
- [ ] No "our previous work" language

### Before Final Submission

- [ ] Abstract registered by January 29, 2026
- [ ] Paper submitted by February 5, 2026
- [ ] PDF validates correctly
- [ ] Conflict of interest declared
- [ ] All co-authors have ORCID IDs
- [ ] Artifact availability statement included

---

## Suggested Paper Title

Based on your work, consider titles like:

1. **"Graph Neural Networks for Interpretable I/O Bottleneck Diagnosis in High-Performance Computing"**

2. **"GNN-IO: Leveraging Job Similarity Graphs for Automated I/O Performance Diagnosis"**

3. **"Beyond Independent Jobs: Graph-Based Machine Learning for HPC I/O Bottleneck Analysis"**

4. **"Attention-Based Graph Neural Networks for Scalable I/O Performance Diagnosis in HPC Systems"**

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

*Good luck with your submission! Feel free to ask if you need clarification on any section.*
