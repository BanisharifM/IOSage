# Detailed References and Resources

Comprehensive research notes for HPDC 2026 I/O Bottleneck Detection project.

---

## Part 1: Competitor Papers (Must-Cite and Compare)

### 1.1 AIIO (HPDC'23) - Primary Baseline

**Paper:** "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis"

**Authors:** Bin Dong, Jean Luca Bez, Suren Byna (Lawrence Berkeley National Laboratory)

**Links:**
- ACM DL: https://dl.acm.org/doi/10.1145/3588195.3592986
- GitHub: https://github.com/hpc-io/aiio
- Contact: dbin@lbl.gov

**What They Did:**
- ML-based I/O bottleneck detection at job level
- Uses multiple models: XGBoost, LightGBM, CatBoost, MLP, TabNet
- SHAP for interpretability
- Trained on Darshan logs from NERSC Cori

**Key Results:**
- 1.8x to 146x I/O performance improvement on test applications
- Works on real applications: E2E, OpenPMD, DASSA

**Limitations (Your Opportunities):**
- Lustre-only (trained on Lustre counters)
- POSIX-focused, limited multi-module
- No graph structure (treats jobs independently)
- No inter-job contention modeling
- Single-label classification (not multi-label)

**How to Use for Comparison:**
- Implement XGBoost/LightGBM yourself with same features
- Compare: Your GNN vs their flat-feature approach
- Show graph structure helps

---

### 1.2 IOAgent (IPDPS'25) - LLM Baseline

**Paper:** "IOAgent: Democratizing Trustworthy HPC I/O Performance Diagnosis Capability via LLMs"

**Authors:** Egersdoerfer C, Sareen A, Bez J, Byna S, Xu D, Dai D

**Links:**
- IEEE: https://ieeexplore.ieee.org/document/11078545/
- OpenReview: https://openreview.net/forum?id=6EwdfIJWzi

**What They Did:**
- LLM-based I/O diagnosis with RAG (Retrieval Augmented Generation)
- Module-based pre-processor for Darshan logs
- Tree-based merger for combining results
- Created TraceBench - labeled I/O trace test suite

**Key Contributions:**
- Works with both proprietary and open-source LLMs
- Interactive interface for follow-up questions
- First comprehensive LLM approach for I/O diagnosis

**Limitations:**
- LLM cost and latency
- No graph structure
- Context window limitations

**How to Use:**
- If you add LLM, compare with IOAgent
- Check if TraceBench dataset is publicly available
- Contact authors for dataset access

---

### 1.3 ION (HotStorage'24) - LLM Approach

**Paper:** "ION: Navigating the HPC I/O Optimization Journey using Large Language Models"

**Authors:** Chris Egersdoerfer, Arnav Sareen, Jean Luca Bez, Suren Byna, Dong Dai

**Links:**
- ACM DL: https://dl.acm.org/doi/10.1145/3655038.3665950
- PDF: https://escholarship.org/content/qt4d4097dx/qt4d4097dx.pdf
- Slides: https://www.hotstorage.org/2024/slides/05_ION_Navigating_the_HPC_IO_Optimization_Journey_using_Large_Language_Models.pdf

**What They Did:**
- LLM with in-context learning for I/O diagnosis
- Chain-of-thought reasoning
- Code generation for optimization suggestions
- Interactive interface

**Key Insight:**
- In-context learning is cost-efficient vs fine-tuning
- Dynamic adjustment to new contexts

**Comparison with IOAgent:**
- ION is earlier (HotStorage workshop)
- IOAgent is more comprehensive (IPDPS main conference)
- Both from same research group

---

### 1.4 Drishti (ISC'23) - Rule-Based Baseline

**Paper:** "Illuminating the I/O Optimization Path of Scientific Applications"

**Authors:** Jean Luca Bez et al.

**Links:**
- Project: https://jeanbez.gitlab.io/isc23/
- GitHub: https://github.com/hpc-io/drishti-io
- Docs: https://drishti-io.readthedocs.io/

**What They Did:**
- Rule-based I/O bottleneck detection
- Provides actionable recommendations
- Multi-module analysis (POSIX, MPI-IO, STDIO)
- Command-line tool + web interface

**Issues Detected:**
- Rank zero heavy workload
- Unbalanced workload across ranks
- Stragglers
- I/O phases
- OST usage issues

**How to Use:**
- Install: `pip install drishti-io`
- Run: `drishti file.darshan`
- Compare your predictions with Drishti output
- Implement similar rules as baseline

---

## Part 2: GNN in HPC (Related but Not I/O)

### 2.1 HPCGCN

**Paper:** "HPCGCN: A Predictive Framework on High Performance Computing Cluster Log Data Using Graph Convolutional Networks"

**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9893918/

**What They Did:**
- GCN for HPC job prediction
- Predicts: job completion/failure, memory/CPU requirements
- Uses user/task data as graph

**Key Point:**
- GNN applied to HPC, but NOT for I/O bottleneck detection
- Proves GNN works for HPC workload analysis
- Your work is first GNN for I/O specifically

---

### 2.2 GNN-RL (SC'24)

**What They Did:**
- GNN + Reinforcement Learning for HPC job scheduling
- Graph models job relationships

**Key Point:**
- Another GNN in HPC, but for scheduling not I/O
- Cite to show GNN is effective in HPC context

---

## Part 3: I/O Analysis Tools

### 3.1 Darshan

**Links:**
- Homepage: https://www.mcs.anl.gov/research/projects/darshan/
- GitHub: https://github.com/darshan-hpc/darshan
- Counter Reference: https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html#_guide_to_darshan_parser_output

**What It Does:**
- Lightweight I/O characterization for HPC
- Captures counters for POSIX, MPI-IO, STDIO, HDF5, etc.
- Runs transparently on production systems
- Generates compressed .darshan log files

**Modules Available:**
- POSIX (70+ counters)
- MPI-IO (30+ counters)
- STDIO (20+ counters)
- HDF5, PnetCDF, Lustre, etc.

---

### 3.2 PyDarshan

**Link:** https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/

**What It Does:**
- Python interface for Darshan logs
- Pandas DataFrame output
- Easy feature extraction

**Usage:**
```python
import darshan
report = darshan.DarshanReport("file.darshan")
posix_df = report.records['POSIX'].to_df()
```

---

### 3.3 DXT Explorer

**Link:** https://github.com/hpc-io/dxt-explorer

**What It Does:**
- Interactive web visualization for DXT traces
- Zoom into I/O operations
- Visual bottleneck identification

**Use Case:**
- Manual validation of your predictions
- Generate figures for paper

---

## Part 4: Benchmarks for Ground Truth

### 4.1 IOR / mdtest

**Link:** https://github.com/hpc/ior

**IOR Features:**
- Synthetic I/O benchmark
- Configurable patterns (sequential, random, strided)
- Multiple interfaces (POSIX, MPI-IO, HDF5)
- Can generate specific bottleneck patterns

**mdtest Features:**
- Metadata-focused benchmark
- Creates/stats/removes files/directories
- Generates metadata-heavy workloads

**Bottleneck Patterns to Generate:**
```bash
# Small I/O
ior -t 64 -b 64K

# Random access
ior -z

# Misaligned
ior -t 1000

# Missing collective (POSIX instead of MPI-IO)
ior -a POSIX

# Metadata heavy
mdtest -n 10000
```

---

### 4.2 DLIO (Critical for AI Workloads)

**Links:**
- GitHub: https://github.com/argonne-lcf/dlio_benchmark
- Docs: https://dlio-benchmark.readthedocs.io/
- Paper: https://ieeexplore.ieee.org/document/9499416/

**What It Does:**
- Emulates deep learning I/O patterns
- Supports TensorFlow, PyTorch data loaders
- Multiple formats: HDF5, TFRecord, NPZ, CSV

**Why Critical for You:**
- Polaris has 4x A100 GPUs per node
- Many workloads are AI/ML training
- DLIO captures these patterns accurately

**Workloads:**
```bash
dlio_benchmark workload=unet3d    # Checkpoint heavy
dlio_benchmark workload=resnet50  # Data loading
dlio_benchmark workload=bert      # NLP patterns
```

**Key Finding:**
- 94% correlation with real ML applications
- MLPerf Storage uses DLIO

---

### 4.3 h5bench

**Links:**
- GitHub: https://github.com/hpc-io/h5bench
- Docs: https://h5bench.readthedocs.io/
- Paper: https://onlinelibrary.wiley.com/doi/10.1002/cpe.8046

**What It Does:**
- HDF5-specific I/O benchmark
- Tests sync vs async I/O
- Multiple access patterns (1D, 2D, 3D)

**Patterns:**
- CONTIG (contiguous)
- INTERLEAVED (array of structures)
- STRIDED

---

### 4.4 IO500

**Links:**
- Website: https://io500.org/
- Lustre Wiki: https://wiki.lustre.org/IOR

**What It Does:**
- Comprehensive storage benchmark
- Combines IOR + mdtest
- Measures bandwidth + IOPS

**Tests:**
- IOR Easy (optimized patterns)
- IOR Hard (challenging patterns)
- mdtest Easy/Hard

---

### 4.5 E3SM-IO

**Link:** https://github.com/Parallel-NetCDF/E3SM-IO

**What It Does:**
- I/O kernel from E3SM climate model
- Non-contiguous, small I/O patterns
- Scientific application representative

**Patterns:**
- Many small, non-contiguous writes
- Cubed sphere grid decomposition
- PnetCDF, HDF5 backends

---

## Part 5: ML Evaluation Without Ground Truth

### 5.1 Sample Size Requirements

**Paper:** "Predicting sample size required for classification performance"

**Link:** https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8

**Key Findings:**
- 75-100 samples typically needed for testing
- Learning curves follow inverse power law
- With nested CV, need 50% fewer samples

**For Your Work:**
- Aim for 100-200 samples per bottleneck type
- Use nested cross-validation
- Total: 2,000-3,000 benchmark logs

---

### 5.2 Evaluating Explanations Without Ground Truth

**Paper:** "Evaluating Explanation Without Ground Truth in Interpretable Machine Learning"

**Link:** https://arxiv.org/abs/1907.06831

**Key Insight:**
- Define explanation quality via predictive accuracy
- Good explanation = identifies features most predictive of model behavior
- Use removal-based evaluation (mask features, check prediction change)

---

### 5.3 LLM-as-Judge

**Paper:** "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods"

**Link:** https://arxiv.org/abs/2412.05579

**Key Findings:**
- LLM evaluations achieve 80%+ agreement with humans
- Can use for explanation quality assessment
- Multiple LLMs (Panel of LLMs) more robust than single

**For Your Work (if using LLM):**
- Use GPT-4 or similar to rate explanation quality
- Self-consistency checking
- Factual consistency with Darshan metrics

---

## Part 6: Clustering Evaluation

### 6.1 Silhouette Score

**Link:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

**What It Does:**
- Measures cluster quality without ground truth
- Range: -1 to 1 (higher = better)
- > 0.5 is "reasonable", > 0.7 is "strong"

**Usage:**
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(features, cluster_labels)
```

---

## Part 7: System Information

### 7.1 ALCF Polaris

**Links:**
- Overview: https://www.alcf.anl.gov/polaris
- User Guide: https://docs.alcf.anl.gov/polaris/

**Specs:**
- 560 nodes (520 compute + 40 other)
- 4x NVIDIA A100 GPUs per node
- AMD EPYC Milan 7543P CPU (32 cores)
- 512 GB DDR4 RAM per node
- 2x Slingshot 11 network adapters

**Storage:**
- Eagle/Grand Lustre: 100 PiB, 650 GiB/s
- Local NVMe: 3.2 TB per node

**Workloads:**
- ~40-50% AI/ML training
- ~30-40% Traditional HPC
- ~10-20% Data analytics

---

## Part 8: Conference Information

### 8.1 HPDC

**Link:** https://hpdc.sci.utah.edu/2025/calls-cfp.html

**Key Requirements:**
- 11 pages (excluding references)
- ACM Proceedings format
- Dual-anonymous review
- Experimental methodology clearly specified
- Limitations acknowledged

**Topics Relevant to Your Work:**
- Storage, I/O, and data management
- Performance modeling, benchmarking, engineering
- AI topics relating to parallel and distributed computing

**Acceptance Rate:** ~18-20% (highly selective)

---

## Part 9: I/O Bottleneck Patterns (From Literature)

### Data Transfer Issues
1. **Small I/O** - Transfer size < 1KB
2. **Random Access** - Non-sequential file access
3. **Misaligned Access** - Not aligned to stripe boundary

### Metadata Issues
4. **Metadata Dominance** - Metadata time > data time
5. **Excessive Opens** - Too many open/close operations

### Parallelism Issues
6. **Load Imbalance** - Uneven distribution across ranks
7. **Missing Collective** - Independent I/O instead of collective
8. **Stragglers** - Some ranks much slower than others

### System Issues
9. **Contention** - Shared resource competition
10. **OST Imbalance** - Uneven Lustre OST usage

### AI/ML Specific
11. **Checkpoint Bottleneck** - Large periodic model saves
12. **Data Loading Slow** - Training data I/O

### Healthy
13. **No Bottleneck** - Efficient I/O patterns

---

## Part 10: Code Repositories

| Tool | GitHub |
|------|--------|
| AIIO | https://github.com/hpc-io/aiio |
| Drishti | https://github.com/hpc-io/drishti-io |
| DXT Explorer | https://github.com/hpc-io/dxt-explorer |
| Darshan | https://github.com/darshan-hpc/darshan |
| IOR | https://github.com/hpc/ior |
| DLIO | https://github.com/argonne-lcf/dlio_benchmark |
| h5bench | https://github.com/hpc-io/h5bench |
| E3SM-IO | https://github.com/Parallel-NetCDF/E3SM-IO |
| PyTorch Geometric | https://github.com/pyg-team/pytorch_geometric |

---

*Last Updated: January 10, 2026*
