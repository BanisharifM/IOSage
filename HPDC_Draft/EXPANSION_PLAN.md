# Plan: HPDC 2026 Paper Expansion to 11 Pages

## Overview

Expand the current 6-page paper to exactly 11 pages with full content, 30-40 references, and complete appendix.

## Key Changes (Jan 10, 2026 - Post Reviewer Feedback)

Based on Claude and Gemini feedback:
1. **Bottleneck classes reduced from 13 to 8** (merged similar classes)
2. **Removed "temporal edges"** - replaced with structural edges + concurrency weights
3. **Removed "First GNN" claim** - replaced with "Topological Resource Contention" formalization
4. **Added Attention-Based Root Cause Localization** as the killer feature
5. **Addressed aggregated log limitation** explicitly in methodology
6. **Self-Supervised Pre-training on 1.37M logs** - Foundation model approach via masked edge reconstruction (addresses distribution shift between benchmarks and production)

## Current State Analysis

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Pages | 6 | 11 | +5 pages |
| Words | ~3,569 | ~6,500-7,500 | +3,000-4,000 words |
| References | 24 | 35-40 | +11-16 references |
| Figures | 3 (placeholder) | 6-8 | +3-5 figures |
| Tables | 5 (placeholder) | 6-8 | +1-3 tables |

## Files to Modify

1. **`paper/main.tex`** - Main paper expansion
2. **`paper/references.bib`** - Add 15+ new references
3. **`paper/appendix.tex`** (NEW) - Create appendix

## Section-by-Section Expansion Plan

### 1. Abstract (~200 words) - KEEP AS-IS
- Already good length and content

### 2. Introduction (Current: 642 words → Target: 900 words, +258 words)
- Add concrete I/O bottleneck statistics (% of runtime lost)
- Expand "why this matters" with specific HPC examples
- Add more detail to "Experimental Methodology" subsection
- Add paper organization paragraph at end

### 3. Background & Related Work (Current: 356 words → Target: 900 words, +544 words)
- **2.1 Darshan I/O Profiling** - Add counter types, module details (+100 words)
- **2.2 Rule-Based I/O Analysis** - Detail Drishti heuristics, WisIO approach (+150 words)
- **2.3 Machine Learning for I/O Analysis** - Expand AIIO details, SHAP (+100 words)
- **2.4 LLM-Based I/O Analysis** - Detail ION/IOAgent architectures (+100 words)
- **2.5 Graph Neural Networks in HPC** - More GNN applications, why suitable (+100 words)

### 4. IOGraphNet Approach (Current: 733 words → Target: 1,100 words, +367 words)
- **3.1 Problem Formulation** - Add formal notation, definitions (+50 words)
- **3.2 Bottleneck Taxonomy** - Expand each category with detection criteria (+100 words)
- **3.3 Graph Construction** - Add algorithm pseudocode box (+100 words)
- **3.4 Model Architecture** - Add hyperparameter justifications, why GAT over GCN (+70 words)
- **3.5 Interpretation via Attention** - Expand with concrete example (+50 words)
- **3.6 NEW: Training Procedure** - Loss function, optimization details (+100 words)

### 5. Evaluation (Current: 986 words → Target: 1,500 words, +514 words)
- **4.2 Datasets** - More detail on Polaris characteristics (+100 words)
- **4.3 Baselines** - Expand baseline descriptions (+80 words)
- **4.5 Implementation Details** - Add compute resources, training time (+50 words)
- **4.6 Main Results** - Expand interpretation of results (+100 words)
- **4.7 Ablation Study** - Add per-edge-type analysis (+80 words)
- **4.9 Per-Class Performance** - Discuss why some classes harder (+50 words)
- **4.10 Generalization** - More analysis of Polaris patterns (+100 words)
- **4.11 NEW: Computational Overhead** - Inference time, scalability (+100 words)

### 6. Discussion (Current: 280 words → Target: 800 words, +520 words)
- **5.1 Why Graph Structure Helps** - Expand with specific examples (+100 words)
- **5.2 Comparison with LLM-Based Approaches** - Add table, deeper analysis (+150 words)
- **5.3 Limitations** - Expand with honest discussion (+100 words)
- **5.4 NEW: Implications for Practitioners** - How to use in practice (+100 words)
- **5.5 NEW: Future Work** - Concrete next steps (+70 words)

### 7. Additional Related Work (Current: 91 words → Target: 200 words, +109 words)
- Add I/O performance prediction papers
- Add more GNN application papers

### 8. Conclusion (Current: 103 words → Target: 150 words, +47 words)
- Summarize key findings more explicitly
- Add future directions

## New Figures to Add

| Figure | Description | Location |
|--------|-------------|----------|
| Fig 4 | Training curves (loss, F1 vs epoch) | Section 4.6 |
| Fig 5 | Confusion matrix heatmap | Section 4.9 |
| Fig 6 | Attention weights visualization | Section 5.1 |
| Fig 7 | Inference time vs graph size | Section 4.11 |

## New Tables to Add

| Table | Description | Location |
|-------|-------------|----------|
| Table 6 | IOGraphNet vs LLM approaches comparison | Section 5.2 |
| Table 7 | Computational overhead comparison | Section 4.11 |

## References to Add (~15 new)

### HPC I/O Papers
- UMAMI (PDSW-DISCS 2017) - Holistic I/O analysis
- Zoom-in Analysis (CCGrid 2019) - I/O log analysis
- HPC I/O Throughput (SC20) - Bottleneck analysis
- I/O Patterns Survey (ACM CSUR 2024) - Comprehensive survey
- Recorder (IPDPS 2020) - Multi-level I/O tracing

### GNN Papers
- GCN (ICLR 2017) - Original graph convolution by Kipf & Welling
- GraphSAGE (NeurIPS 2017) - Inductive learning on graphs
- GIN (ICLR 2019) - Weisfeiler-Lehman isomorphism
- GNNExplainer (NeurIPS 2019) - Interpretation for GNNs

### ML for Systems
- DeepIO (FAST 2020) - Deep learning for I/O
- Learned Index (SIGMOD 2018) - ML for systems paradigm

### Multi-label Classification
- Multi-label survey (TKDE 2014) - Zhang & Zhou
- Evaluation metrics for multi-label (Pattern Recognition 2012)

### Darshan Extended
- DXT paper (CUG 2017) - Extended tracing
- Darshan 3.0 paper (ESPT 2016)

## Appendix Content (NEW FILE: appendix.tex)

### A. Extended Experimental Results
- Full hyperparameter search results table
- Additional ablation studies (learning rate, dropout)
- Per-bottleneck detection examples with attention maps

### B. Dataset Details
- Polaris log statistics table (size distribution, module coverage)
- Benchmark configuration details for each bottleneck type
- Feature normalization procedure

### C. Implementation Details
- Complete feature list (all 120+ Darshan counters used)
- Graph construction algorithm pseudocode
- Model hyperparameters table with justifications

### D. Reproducibility
- Code availability statement (GitHub link placeholder)
- Hardware specifications (GPU, memory)
- Training procedure step-by-step
- Random seed information

## Implementation Order

1. **Phase 1: References** (~15 min)
   - Add 15+ new references to references.bib
   - Verify all existing references are real

2. **Phase 2: Background Expansion** (~30 min)
   - Expand Related Work section from 356 to 900 words
   - Add new citations throughout

3. **Phase 3: Technical Sections** (~45 min)
   - Expand Approach section with algorithm box
   - Expand Evaluation with new subsections
   - Add computational overhead subsection

4. **Phase 4: Discussion** (~30 min)
   - Major expansion from 280 to 800 words
   - Add practitioner implications
   - Add future work

5. **Phase 5: Figures & Tables** (~20 min)
   - Add placeholder boxes for 4 new figures
   - Add 2 new tables
   - Ensure all have \Description{}

6. **Phase 6: Appendix** (~20 min)
   - Create appendix.tex with 4 sections
   - Add \input{appendix} to main.tex

7. **Phase 7: Compile & Verify** (~10 min)
   - Compile PDF
   - Verify ~11 pages (excluding appendix)
   - Fix any warnings

## Word Count Targets by Section

| Section | Current | Target | Priority |
|---------|---------|--------|----------|
| Introduction | 642 | 900 | Medium |
| Background | 356 | 900 | **HIGH** |
| Approach | 733 | 1,100 | Medium |
| Evaluation | 986 | 1,500 | **HIGH** |
| Discussion | 280 | 800 | **HIGH** |
| Related Work | 91 | 200 | Low |
| Conclusion | 103 | 150 | Low |
| **TOTAL TEXT** | ~3,191 | ~5,550 | |
| +Abstract/Front | ~400 | ~400 | |
| +Figs/Tables | ~1 page | ~2 pages | |
| **FINAL** | 6 pages | 11 pages | |

## Key Content Sources (from project .md files)

### From HPDC2026_Revised_Roadmap.md:
- "Why Graph?" examples for Discussion section
- **8 bottleneck taxonomy** (consolidated from original 13)
- GAT architecture with attention-based root cause localization
- Topological Resource Contention framing
- Reviewer response strategies

### From PROJECT_PHASES.md:
- All Darshan counters for Appendix
- Benchmark scripts for evaluation
- Derived metrics calculations
- Graph construction code

### From IO_Bottleneck_Detection_Guide.md:
- Complete counter reference for Background
- Bottleneck detection patterns
- UMAMI, Zoom-in citations

### From REFERENCES_LIST.md:
- Additional paper URLs
- Tool documentation links

## Quality Checklist

Before completion:
- [ ] Paper is exactly ~11 pages (excluding refs/appendix)
- [ ] All 35-40 references verified and real
- [ ] All figures have \Description{} for accessibility
- [ ] All tables fit within column width (use \resizebox if needed)
- [ ] No Overfull hbox LaTeX warnings
- [ ] Claims are conservative and grounded in existing work
- [ ] Appendix follows HPDC guidelines
- [ ] Anonymous for review (no author names, no identifying info)
- [ ] CCS concepts and keywords present
- [ ] ACM reference format correct

## HPDC 2026 Deadlines (Reminder)
- **Abstract deadline:** January 29, 2026
- **Paper deadline:** February 5, 2026
- **Max pages:** 11 (excluding references)
- **Format:** ACM sigconf, double-column
