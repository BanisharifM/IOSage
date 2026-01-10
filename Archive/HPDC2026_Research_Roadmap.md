# HPDC 2026 Research Roadmap: Novel AI/GNN Approaches for I/O Performance Bottleneck Detection

## Executive Summary

This document provides a comprehensive research roadmap for developing a novel I/O performance bottleneck detection system targeting **ACM HPDC 2026**. Based on extensive literature review and gap analysis, we propose **two complementary approaches**:

1. **IOGraphNet**: A Graph Neural Network approach for modeling job relationships and detecting I/O bottlenecks
2. **IOExplainLLM**: An LLM-augmented approach for interpretable bottleneck diagnosis and recommendations

Both approaches leverage the unique characteristics of the **ALCF Polaris Darshan Log Collection** (1.37M+ logs) and address critical gaps in existing research.

---

## Part 1: HPDC Conference Analysis

### 1.1 Conference Profile

| Attribute | Details |
|-----------|---------|
| **Full Name** | ACM International Symposium on High-Performance Parallel and Distributed Computing |
| **Acceptance Rate** | ~18-20% (highly selective) |
| **Review Process** | Dual-anonymous (double-blind) |
| **Paper Length** | 11 pages (excluding references) |
| **Typical Deadline** | Early February |
| **Notification** | Early April |

### 1.2 Topics of Interest (Relevant to Our Work)

From the [HPDC 2025 CFP](https://hpdc.sci.utah.edu/2025/calls-cfp.html):

- **Storage, I/O, and data management** - DIRECTLY RELEVANT
- **Performance modeling, benchmarking, and engineering** - DIRECTLY RELEVANT
- **Scientific applications, algorithms, and workflows** - RELEVANT
- **Resource management and scheduling** - RELEVANT
- **AI topics relating to parallel and distributed computing** - DIRECTLY RELEVANT

### 1.3 Evaluation Criteria

HPDC papers are evaluated on:

1. **Novelty** - New ideas or significant extensions
2. **Scientific Value** - Rigorous methodology and results
3. **Demonstrated Usefulness** - Real-world applicability
4. **Potential Impact** - Contribution to the field
5. **Quality of Presentation** - Clear motivation and articulation

### 1.4 What Makes a Strong HPDC Paper

Based on CFP and past papers:

- Clear motivation with **quantitative evidence**
- Articulation of **prior work limitations**
- Specific **key insights** advancing the field
- **Transparent experimental methodology**
- Honest acknowledgment of **approach limitations**
- **Reproducibility** (code/data availability is valued)

---

## Part 2: Literature Review and Gap Analysis

### 2.1 Existing AI/ML Approaches for I/O Bottleneck Detection

#### AIIO (HPDC'23) - Current State-of-the-Art

| Aspect | Details |
|--------|---------|
| **Authors** | Bin Dong, Jean Luca Bez, Suren Byna (Berkeley Lab) |
| **Approach** | ML classification for job-level bottleneck diagnosis |
| **Data Source** | Darshan logs from NERSC Cori |
| **Results** | 1.8x-146x improvement on test applications |
| **GitHub** | https://github.com/hpc-io/aiio |

**AIIO Limitations (Our Opportunities):**

1. **Lustre-specific** - Models trained only on Lustre counters
2. **Single module focus** - Primarily POSIX, limited multi-layer analysis
3. **No job correlation** - Treats each job independently
4. **Limited interpretability** - Black-box classification
5. **No temporal modeling** - Ignores time-series patterns
6. **No cross-system generalization** - Tied to Cori system

#### Other ML Approaches

| Paper | Approach | Gap |
|-------|----------|-----|
| I/O Burst Prediction (2023) | XGBoost for burst forecasting | System-level only, not job-level |
| Gauge (SC20) | Explainable models for throughput | No multi-module, no GNN |
| tf-Darshan | TensorFlow I/O profiling | Specific to ML workloads |

### 2.2 GNN in HPC Systems (NOT for I/O Bottleneck Detection)

| System | Application | I/O Focus? |
|--------|-------------|------------|
| GNN-RL (SC24) | Job scheduling | NO |
| HPCGCN | Job completion prediction | NO |
| GrapheonRL | Workflow scheduling | NO |
| TARDIS | Power-aware scheduling | NO |

**Critical Gap: NO existing work uses GNN for I/O bottleneck detection!**

### 2.3 Multi-Module Analysis Gap

Darshan captures data from multiple modules:
- POSIX, MPI-IO, STDIO (core)
- HDF5, PnetCDF (high-level)
- Lustre, GPFS (file system specific)
- DXT (extended tracing)
- APMPI, HEATMAP (auxiliary)

**Gap**: Existing approaches analyze modules in isolation. No work models **cross-module relationships and propagation of bottlenecks across the I/O stack**.

### 2.4 Summary of Research Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **G1** | No GNN for I/O bottleneck detection | Major novelty opportunity |
| **G2** | No multi-module correlation modeling | Misses cross-layer bottlenecks |
| **G3** | No job-to-job interference modeling | Ignores contention effects |
| **G4** | Limited interpretability | Users can't understand why |
| **G5** | No temporal pattern analysis | Misses time-varying behaviors |
| **G6** | Single-system training | Poor generalization |

---

## Part 3: Your Dataset - ALCF Polaris Darshan Logs

### 3.1 Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Source** | ALCF Polaris Supercomputer |
| **Logs** | 1,378,117 .darshan files |
| **Time Span** | April 2024 - January 2026 |
| **Total Size** | ~17.8 GB (compressed) |
| **DOI** | 10.5281/zenodo.15052603 |

### 3.2 System Specifications (Polaris)

| Component | Specification |
|-----------|---------------|
| Nodes | 520 HPE Apollo 6500 Gen 10+ |
| CPU | AMD EPYC Milan 7543P (32-core) |
| GPU | 4x NVIDIA A100 per node |
| Storage | Eagle/Grand Lustre (100 PiB, 650 GiB/s) |
| Network | 2x Slingshot 11 |

### 3.3 Available Modules in Dataset

Based on sample analysis, logs contain:
- **POSIX** - Core file operations
- **MPI-IO** - Parallel I/O
- **STDIO** - Stream I/O
- **APMPI** - MPI profiling
- **HEATMAP** - Access pattern visualization

### 3.4 Dataset Advantages for Research

1. **Scale**: 1.37M+ jobs provides statistical power
2. **Recency**: 2024-2026 data reflects modern workloads
3. **GPU Workloads**: A100 GPUs mean AI/ML workloads present
4. **Multi-Module**: Rich instrumentation data
5. **Production Data**: Real scientific applications
6. **Anonymized**: Publishable without privacy concerns

---

## Part 4: Proposed Approach 1 - IOGraphNet (GNN-Based)

### 4.1 Core Novelty

**First GNN-based approach for I/O performance bottleneck detection that models:**
1. Multi-module relationships within a job
2. Job-to-job interference through temporal co-location
3. File system resource contention patterns

### 4.2 Graph Construction

#### 4.2.1 Intra-Job Module Graph

For each job, construct a heterogeneous graph:

```
Nodes:
- POSIX file records
- MPI-IO file records
- STDIO file records
- HDF5 dataset records (if present)

Edges:
- Same-file edges (POSIX ↔ MPI-IO for same file)
- Call-stack edges (HDF5 → MPI-IO → POSIX)
- Temporal edges (operations ordered by timestamp)
```

Node Features (per file record):
```python
features = {
    # Volume metrics
    'bytes_read': POSIX_BYTES_READ,
    'bytes_written': POSIX_BYTES_WRITTEN,

    # Operation counts
    'reads': POSIX_READS,
    'writes': POSIX_WRITES,
    'opens': POSIX_OPENS,

    # Access patterns
    'seq_reads': POSIX_SEQ_READS,
    'consec_reads': POSIX_CONSEC_READS,
    'rw_switches': POSIX_RW_SWITCHES,

    # Size distribution (histogram)
    'size_0_100': POSIX_SIZE_READ_0_100,
    'size_100_1k': POSIX_SIZE_READ_100_1K,
    # ... all size buckets

    # Alignment
    'file_not_aligned': POSIX_FILE_NOT_ALIGNED,
    'mem_not_aligned': POSIX_MEM_NOT_ALIGNED,

    # Timing
    'read_time': POSIX_F_READ_TIME,
    'write_time': POSIX_F_WRITE_TIME,
    'meta_time': POSIX_F_META_TIME,

    # Variance (for shared files)
    'rank_time_variance': POSIX_F_VARIANCE_RANK_TIME,
    'rank_bytes_variance': POSIX_F_VARIANCE_RANK_BYTES,
}
```

#### 4.2.2 Inter-Job Contention Graph

For jobs running in overlapping time windows:

```
Nodes:
- Job nodes (aggregated features from intra-job graph)

Edges:
- Temporal overlap edges (jobs running simultaneously)
- Resource sharing edges (same Lustre OSTs)
- User edges (same anonymized user ID)
```

### 4.3 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IOGraphNet Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Darshan Log → Graph Construction                    │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Intra-Job GNN (Heterogeneous Message Passing)      │   │
│  │  - GraphSAGE / GAT layers                           │   │
│  │  - Module-specific encoders                         │   │
│  │  - Cross-module attention                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Temporal Aggregation                                │   │
│  │  - LSTM / Transformer for time-series               │   │
│  │  - Attention over operation sequences               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Inter-Job GNN (Contention Modeling)                │   │
│  │  - Temporal graph convolution                       │   │
│  │  - Resource contention edges                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Multi-Task Output Heads                             │   │
│  │  - Bottleneck Classification (7 types)              │   │
│  │  - Performance Prediction (bandwidth)               │   │
│  │  - Severity Scoring                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Bottleneck Classification Labels

Define 7 bottleneck types based on literature:

| ID | Bottleneck Type | Detection Criteria |
|----|-----------------|-------------------|
| B1 | Small I/O | SIZE_0_100 + SIZE_100_1K > 80% of ops |
| B2 | Metadata Dominance | META_TIME > READ_TIME + WRITE_TIME |
| B3 | Load Imbalance | VARIANCE_RANK_TIME > threshold |
| B4 | Missing Collective I/O | MPIIO_INDEP > 0 && MPIIO_COLL == 0 |
| B5 | Misaligned Access | FILE_NOT_ALIGNED / total_ops > 0.5 |
| B6 | Random Access | SEQ_READS / READS < 0.3 |
| B7 | Contention | Low bandwidth during high system load |

### 4.5 Training Strategy

1. **Self-Supervised Pre-training**
   - Mask node features and predict them
   - Predict edge existence
   - Contrastive learning between similar jobs

2. **Supervised Fine-tuning**
   - Label subset using rule-based heuristics
   - Active learning for uncertain samples

3. **Multi-Task Learning**
   - Joint bottleneck classification + bandwidth prediction
   - Auxiliary task: predict runtime

### 4.6 Novelty Claims

1. **First GNN for I/O bottleneck detection**
2. **Heterogeneous graph modeling multiple I/O modules**
3. **Inter-job contention modeling via temporal graphs**
4. **Multi-task learning for bottleneck type + severity**

---

## Part 5: Proposed Approach 2 - IOExplainLLM (LLM-Augmented)

### 5.1 Core Novelty

**First LLM-augmented system for interpretable I/O bottleneck diagnosis with natural language recommendations.**

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 IOExplainLLM Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Darshan Log                                         │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Feature Extraction & Encoding                       │   │
│  │  - Multi-module feature vectors                      │   │
│  │  - Derived metrics computation                       │   │
│  │  - Pattern detection                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Neural Bottleneck Detector (Transformer)           │   │
│  │  - Self-attention over counter sequences            │   │
│  │  - Cross-module attention                           │   │
│  │  - Bottleneck classification                        │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Explanation Generator (Fine-tuned LLM)             │   │
│  │  - Input: Detected bottlenecks + key metrics        │   │
│  │  - Output: Natural language explanation             │   │
│  │  - Recommendations for optimization                 │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RAG Knowledge Base                                  │   │
│  │  - I/O optimization best practices                  │   │
│  │  - System-specific tuning guides                    │   │
│  │  - Historical case studies                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Key Components

#### 5.3.1 Multi-Head Attention for Counters

```python
class CounterAttention(nn.Module):
    """
    Attention mechanism to identify which counters
    are most relevant for bottleneck detection.
    """
    def __init__(self, num_counters, hidden_dim, num_heads):
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.counter_embedding = nn.Linear(num_counters, hidden_dim)

    def forward(self, counter_values):
        # counter_values: [batch, num_modules, num_counters]
        embedded = self.counter_embedding(counter_values)
        attn_output, attn_weights = self.attention(embedded, embedded, embedded)
        return attn_output, attn_weights  # weights provide interpretability
```

#### 5.3.2 LLM Prompt Template

```
You are an HPC I/O performance expert analyzing a Darshan log.

## Job Summary
- Job ID: {job_id}
- Runtime: {runtime} seconds
- Processes: {nprocs}
- Total Read: {bytes_read} bytes
- Total Write: {bytes_written} bytes

## Detected Bottlenecks
{bottleneck_list}

## Key Metrics
- Average read size: {avg_read_size} bytes
- Average write size: {avg_write_size} bytes
- Metadata time ratio: {meta_ratio}%
- Sequential access ratio: {seq_ratio}%
- MPI-IO to POSIX ratio: {mpiio_posix_ratio}

## Task
1. Explain WHY these bottlenecks occurred
2. Provide SPECIFIC recommendations to improve I/O performance
3. Estimate potential performance improvement

## Response Format
### Root Cause Analysis
[Your analysis]

### Recommendations
1. [Specific recommendation 1]
2. [Specific recommendation 2]
...

### Expected Improvement
[Quantitative estimate if possible]
```

### 5.4 Novelty Claims

1. **First LLM-augmented I/O bottleneck diagnosis**
2. **Attention-based interpretability for counter importance**
3. **Natural language recommendations for non-experts**
4. **RAG integration with I/O best practices**

---

## Part 6: Realistic Evaluation Strategy (Critical for HPDC Acceptance)

### 6.0 The Core Challenge

**Your Constraints:**
1. ❌ No ground truth labels for 1.37M production logs
2. ❌ No human experts available for manual labeling
3. ❌ Cannot rerun the production jobs
4. ❌ Need to validate BOTH detection AND explanations
5. ⚠️ HPDC requires rigorous experimental methodology

**The Key Question:** How do we prove ANY ML model (GNN, XGBoost, LLM) is detecting bottlenecks correctly?

### 6.1 Answer to HPDC Requirements

**Does HPDC require performance improvement or just detection?**

Based on [HPDC 2025 CFP](https://hpdc.sci.utah.edu/2025/calls-cfp.html):
- Primary focus is "new research ideas supported by **experimental implementation and evaluation**"
- Must include "experimental methodology and artifact availability"
- Must "support experimental methodology choices by citing relevant prior works"

**Answer:** Detection + Validation is acceptable, but you need to demonstrate **practical usefulness**. AIIO (HPDC'23) showed up to 146x improvement on test applications. You should aim for at least one of:
1. Controlled benchmark validation with known patterns
2. OR intervention study on selected applications
3. OR strong theoretical/analytical validation

### 6.2 Multi-Layer Validation Strategy

Instead of relying on a single validation approach, use **multiple independent validation methods** that collectively provide confidence:

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Layer Validation Framework                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: CONTROLLED BENCHMARKS (Ground Truth Available)       │
│  ├── IOR with intentional bottleneck patterns                  │
│  ├── mdtest for metadata-heavy patterns                        │
│  ├── DLIO for ML workload patterns                             │
│  └── Synthetic jobs with known issues                          │
│           │                                                     │
│           ▼ Validates: Detection accuracy on known patterns     │
│                                                                 │
│  Layer 2: INTERNAL CONSISTENCY CHECKS                          │
│  ├── Cross-module correlation validation                       │
│  ├── Physical constraints checking                             │
│  ├── Self-consistency of predictions                           │
│  └── Clustering quality metrics (silhouette, modularity)       │
│           │                                                     │
│           ▼ Validates: Model is learning meaningful patterns    │
│                                                                 │
│  Layer 3: ABLATION STUDIES                                     │
│  ├── Remove GNN layers → performance drops?                    │
│  ├── Remove multi-module → less accurate?                      │
│  ├── Random labels → model fails?                              │
│  └── Feature importance matches domain knowledge?              │
│           │                                                     │
│           ▼ Validates: Each component contributes meaningfully  │
│                                                                 │
│  Layer 4: INTERVENTION STUDY (Limited Scale)                   │
│  ├── Run 5-10 benchmarks with detected bottlenecks             │
│  ├── Apply recommended optimizations                           │
│  └── Measure actual performance improvement                    │
│           │                                                     │
│           ▼ Validates: Recommendations lead to real improvement │
│                                                                 │
│  Layer 5: CROSS-METHOD AGREEMENT                               │
│  ├── Compare GNN predictions with rule-based                   │
│  ├── Compare with existing tools (DXT Explorer, AIIO)          │
│  └── High agreement = confidence; disagreement = investigate   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Resolving the "Why Use ML if Rules Verify It?" Paradox

**The Concern:** If rule-based code is the verification source, why use ML?

**The Answer (for reviewers):**

| Aspect | Rule-Based | ML (GNN) | Why ML is Better |
|--------|------------|----------|------------------|
| **Known Patterns** | ✅ Works | ✅ Works | Same performance |
| **Threshold Selection** | Manual tuning | Learned | **ML adapts to data** |
| **Complex Patterns** | ❌ Cannot detect | ✅ Learns | **ML finds non-obvious combinations** |
| **Contention Effects** | ❌ Hard to model | ✅ Graph modeling | **Inter-job relationships** |
| **Generalization** | ❌ System-specific | ✅ Learns features | **Cross-system potential** |
| **Severity Scoring** | ❌ Binary | ✅ Continuous | **Prioritization** |

**Key Insight for Paper:**

> "We validate on benchmark patterns where ground truth is available. The ML model matches rule-based detection on simple patterns (proving correctness) but ALSO detects complex multi-factor bottlenecks that rules miss (proving added value)."

### 6.4 Controlled Benchmark Experiments (Layer 1)

#### Creating Ground Truth with IOR

```bash
# Small I/O bottleneck (B1)
mpirun -np 64 ior -a POSIX -t 64 -b 64K -w -r

# Metadata bottleneck (B2)
mpirun -np 64 mdtest -n 10000 -d /lus/grand/test

# Load imbalance (B3)
# Use custom script with rank-specific delays

# Missing collective (B4)
mpirun -np 64 ior -a POSIX -t 1M -b 1G  # vs MPIIO collective

# Misaligned access (B5)
mpirun -np 64 ior -a POSIX -t 1000 -b 1G  # Non-aligned size

# Random access (B6)
mpirun -np 64 ior -a POSIX -t 1M -b 1G -z  # Random offsets

# Contention (B7)
# Run multiple jobs simultaneously on same OSTs
```

**Evaluation Metrics:**
- Precision/Recall/F1 for each bottleneck type
- Confusion matrix
- Detection latency

#### DLIO Benchmark for ML Workloads

```bash
# Checkpoint bottleneck pattern
mpirun -np 64 dlio_benchmark workload=unet3d

# Data loading bottleneck
mpirun -np 64 dlio_benchmark workload=resnet50
```

### 6.5 Internal Consistency Validation (Layer 2)

#### Physical Constraints Checking

```python
def validate_physical_constraints(predictions, features):
    """
    Check if model predictions violate physical laws
    """
    violations = []

    # Bandwidth cannot exceed theoretical maximum
    if predicted_bandwidth > MAX_LUSTRE_BW:
        violations.append("Exceeds physical bandwidth limit")

    # Read time must be >= bytes_read / max_bandwidth
    min_read_time = features['bytes_read'] / MAX_LUSTRE_BW
    if predicted_read_time < min_read_time:
        violations.append("Read time below physical minimum")

    # If no writes, cannot have write bottleneck
    if features['writes'] == 0 and predicts_write_bottleneck:
        violations.append("Write bottleneck without writes")

    return len(violations) == 0
```

#### Clustering Quality Metrics (No Ground Truth Needed)

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Cluster jobs by bottleneck type
labels = model.predict_bottleneck_types(all_jobs)

# Silhouette score: -1 to 1 (higher = better clusters)
silhouette = silhouette_score(job_features, labels)
# > 0.5 is "reasonable", > 0.7 is "strong"

# Calinski-Harabasz: higher = better
ch_score = calinski_harabasz_score(job_features, labels)
```

### 6.6 Ablation Studies (Layer 3)

| Experiment | What to Measure | Expected Result |
|------------|-----------------|-----------------|
| Remove GNN, use MLP | Accuracy drops | GNN structure matters |
| Remove inter-job edges | Contention detection drops | Job relationships matter |
| POSIX only vs Multi-module | F1 drops | Multi-layer analysis helps |
| Random feature shuffle | Accuracy drops to baseline | Model learns real patterns |
| Remove attention | Interpretability drops | Attention aids understanding |
| Permutation importance | Top features match domain | Model aligns with theory |

### 6.7 Limited Intervention Study (Layer 4)

**What's Feasible:**
- Run 5-10 I/O benchmark applications (IOR, DLIO, HACC-IO)
- Let model detect bottlenecks and suggest optimizations
- Apply optimizations and rerun
- Measure before/after performance

**NOT Required:**
- Rerunning 1.37M production jobs (impossible)
- Human expert validation (not available)
- Modifying real user applications

**Sample Results Table for Paper:**

| Benchmark | Detected Bottleneck | Recommendation | Before | After | Improvement |
|-----------|---------------------|----------------|--------|-------|-------------|
| IOR-small | Small I/O (B1) | Increase block size | 50 MB/s | 2 GB/s | 40x |
| mdtest-meta | Metadata (B2) | Reduce file count | 1000 ops/s | 5000 ops/s | 5x |
| DLIO-ckpt | Collective missing | Use MPIIO | 200 MB/s | 1.5 GB/s | 7.5x |

### 6.8 Cross-Method Agreement (Layer 5)

```python
def cross_validate_methods(job):
    """Compare multiple detection methods"""

    # 1. Rule-based detection
    rule_result = rule_based_detector.detect(job)

    # 2. Our GNN model
    gnn_result = gnn_model.predict(job)

    # 3. AIIO (if available)
    aiio_result = aiio_model.predict(job)

    # 4. DXT Explorer heuristics
    dxt_result = dxt_heuristics.detect(job)

    # Agreement analysis
    agreement = {
        'all_agree': rule == gnn == aiio == dxt,
        'gnn_vs_rule': gnn == rule,
        'gnn_vs_aiio': gnn == aiio,
    }

    return agreement
```

**For Paper:**
- Report % agreement between methods
- High agreement on simple patterns (validates correctness)
- Analyze disagreements (potential novel detections)

### 6.9 LLM Explanation Validation

For LLM-generated explanations, use multiple validation approaches:

#### A. Factual Consistency Checking

```python
def check_factual_consistency(explanation, job_metrics):
    """
    Verify LLM claims against actual metrics
    """
    claims = extract_claims(explanation)

    for claim in claims:
        if claim.type == "numeric":
            # "The job performed 1.5M reads"
            actual = job_metrics[claim.metric]
            if abs(claim.value - actual) / actual > 0.1:  # 10% tolerance
                return False, f"Incorrect: {claim}"

        if claim.type == "comparative":
            # "Read time exceeded write time"
            if not eval_comparison(claim, job_metrics):
                return False, f"Incorrect comparison: {claim}"

    return True, "All claims verified"
```

#### B. LLM-as-Judge Evaluation

Use a separate LLM to evaluate explanation quality:

```python
def llm_judge_evaluation(explanation, job_context):
    prompt = f"""
    You are evaluating an I/O bottleneck explanation.

    Job Context:
    {job_context}

    Explanation to evaluate:
    {explanation}

    Rate on these criteria (1-5):
    1. Accuracy: Does it correctly identify the bottleneck?
    2. Clarity: Is the explanation understandable?
    3. Actionability: Are the recommendations specific and feasible?
    4. Completeness: Does it address all major issues?

    Return JSON with scores and justification.
    """
    return judge_llm.evaluate(prompt)
```

#### C. Self-Consistency Test

```python
def self_consistency_test(job, n_samples=5):
    """
    Run same job through LLM multiple times
    Check if explanations are consistent
    """
    explanations = [generate_explanation(job) for _ in range(n_samples)]

    # Extract key claims from each
    all_claims = [extract_key_claims(e) for e in explanations]

    # Measure overlap
    consistency = measure_claim_overlap(all_claims)

    return consistency  # Should be > 0.8 for reliable explanations
```

### 6.10 Recommended Evaluation for Your Paper

Given your constraints, here's the practical evaluation approach:

**Primary Evaluation (Must Include):**

1. **Controlled Benchmarks (30% of evaluation)**
   - IOR with 7 bottleneck patterns
   - mdtest for metadata
   - DLIO for ML workloads
   - Report precision/recall/F1 per pattern

2. **Ablation Studies (25% of evaluation)**
   - Prove each component matters
   - Compare GNN vs MLP vs XGBoost
   - Multi-module vs POSIX-only

3. **Clustering Quality (15% of evaluation)**
   - Silhouette scores
   - Cluster visualization
   - Feature importance analysis

**Secondary Evaluation (Strengthens Paper):**

4. **Limited Intervention (15% of evaluation)**
   - 5-10 benchmarks with before/after
   - Show recommendations work

5. **Cross-Method Agreement (10% of evaluation)**
   - Agreement with rule-based
   - Novel detections analysis

6. **Scalability Analysis (5% of evaluation)**
   - Training time vs dataset size
   - Inference time per job

### 6.11 Comparison with AIIO Evaluation

| Aspect | AIIO (HPDC'23) | Your Approach |
|--------|----------------|---------------|
| Training Data | Cori Darshan logs | Polaris Darshan logs |
| Ground Truth | Rule-based labels | Rule-based + benchmark |
| Validation | 5 real applications | IOR/DLIO + intervention |
| Performance Test | Yes (up to 146x) | Limited intervention |
| Interpretability | SHAP values | GNN attention + LLM |
| Multi-Module | Limited | Full stack |

**Your Advantage:** Multi-layer validation framework is more rigorous than AIIO's single-method validation.

### 6.12 Existing Tools for Validation

| Tool | Purpose | How to Use |
|------|---------|-----------|
| [DXT Explorer](https://github.com/hpc-io/dxt-explorer) | Visual bottleneck detection | Cross-reference predictions |
| [AIIO](https://github.com/hpc-io/aiio) | ML bottleneck detection | Baseline comparison |
| [darshan-util](https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html) | Log parsing/summary | Feature extraction |
| [PyDarshan](https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/index.html) | Python analysis | Data processing |

### 6.13 Datasets

#### Training Data
- **ALCF Polaris Logs**: 1.37M logs (your dataset)
- Split: 70% train, 15% validation, 15% test

#### Evaluation Benchmarks

| Benchmark | Description | Purpose |
|-----------|-------------|---------|
| **IOR** | Synthetic I/O benchmark | Controlled patterns |
| **mdtest** | Metadata benchmark | Metadata bottlenecks |
| **DLIO** | Deep learning I/O | ML workload patterns |
| **HACC-IO** | Cosmology I/O kernel | Real app pattern |
| **MADBench2** | CMB analysis kernel | Scientific workload |

### 6.14 Baselines

| Baseline | Description |
|----------|-------------|
| **AIIO** | State-of-the-art ML approach |
| **Rule-based** | Threshold-based heuristics |
| **Random Forest** | Classical ML baseline |
| **XGBoost** | Gradient boosting baseline |
| **MLP** | Deep learning baseline (no graph structure) |

### 6.15 Ablation Studies Summary

| Component | Ablation |
|-----------|----------|
| Intra-job GNN | Remove, use flat features |
| Inter-job GNN | Remove contention modeling |
| Temporal attention | Remove, use mean pooling |
| Multi-module | POSIX only vs all modules |
| LLM explanations | Remove, show raw predictions |

---

## Part 7: Implementation Roadmap

### 7.1 Phase 1: Data Preparation (2-3 weeks)

```
Week 1-2:
├── Parse all Darshan logs with PyDarshan
├── Extract features from all modules
├── Build feature database (SQLite/Parquet)
├── Compute derived metrics
└── Generate bottleneck labels (rule-based)

Week 3:
├── Graph construction pipeline
├── Temporal alignment of jobs
├── Train/val/test split
└── Data quality analysis
```

### 7.2 Phase 2: Model Development (4-6 weeks)

```
Week 4-5:
├── Implement intra-job GNN
├── Module-specific encoders
├── Cross-module attention
└── Initial training experiments

Week 6-7:
├── Implement inter-job contention graph
├── Temporal graph convolution
├── Multi-task output heads
└── Hyperparameter tuning

Week 8-9:
├── LLM integration (if doing Approach 2)
├── Prompt engineering
├── RAG knowledge base construction
└── Fine-tuning experiments
```

### 7.3 Phase 3: Evaluation (3-4 weeks)

```
Week 10-11:
├── Run benchmark applications
├── Collect Darshan logs from benchmarks
├── Evaluate on test set
├── Compare with baselines

Week 12-13:
├── Ablation studies
├── Generalization experiments
├── Interpretability evaluation
└── End-to-end improvement validation
```

### 7.4 Phase 4: Paper Writing (3-4 weeks)

```
Week 14-15:
├── Draft introduction and related work
├── Method section with figures
├── Results and analysis
└── Internal review

Week 16-17:
├── Revisions based on feedback
├── Polish figures and tables
├── Prepare supplementary materials
├── Final proofreading
└── Submit to HPDC 2026
```

---

## Part 8: Novelty Summary and Contribution Claims

### 8.1 IOGraphNet Contributions

| # | Contribution | Novelty Level |
|---|--------------|---------------|
| C1 | First GNN-based I/O bottleneck detection | **HIGH** - No prior work |
| C2 | Heterogeneous graph for multi-module modeling | **HIGH** - Novel representation |
| C3 | Inter-job contention via temporal graphs | **MEDIUM-HIGH** - New application |
| C4 | Multi-task bottleneck + performance prediction | **MEDIUM** - Standard but effective |

### 8.2 IOExplainLLM Contributions

| # | Contribution | Novelty Level |
|---|--------------|---------------|
| C1 | First LLM-augmented I/O diagnosis | **HIGH** - No prior work |
| C2 | Attention-based counter interpretability | **MEDIUM-HIGH** - Novel application |
| C3 | Natural language recommendations | **HIGH** - Practical impact |
| C4 | RAG with I/O best practices | **MEDIUM** - Engineering contribution |

### 8.3 Shared Contributions

| # | Contribution | Novelty Level |
|---|--------------|---------------|
| C1 | Multi-module analysis (not just POSIX) | **MEDIUM-HIGH** - Addresses clear gap |
| C2 | Comprehensive evaluation on real production data | **HIGH** - 1.37M logs |
| C3 | Generalization to unseen benchmarks | **HIGH** - Practical validation |
| C4 | Open-source release | **VALUE-ADD** - Reproducibility |

---

## Part 9: Risk Analysis and Mitigation

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GNN doesn't outperform baselines | Medium | High | Ensure strong baselines, have backup features |
| Labeling is inaccurate | Medium | High | Validate labels with domain experts |
| Poor generalization | Medium | High | Cross-validation, diverse benchmarks |
| LLM hallucinations | Low | Medium | RAG grounding, output validation |

### 9.2 Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data preprocessing takes longer | High | Medium | Start early, parallelize |
| Model training is slow | Medium | Medium | Use GPU cluster, efficient batching |
| Paper writing delays | Medium | High | Start writing early, outline first |

### 9.3 Review Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Novelty questioned | Medium | High | Clear positioning against AIIO |
| Evaluation deemed insufficient | Medium | High | Comprehensive experiments |
| Reproducibility concerns | Low | Medium | Release code and data pointers |

---

## Part 10: Recommended Approach Selection

### For Your Master's Thesis (GNN Focus)

**Recommended: IOGraphNet (Approach 1)**

Reasons:
1. Clear GNN novelty for thesis
2. Strong technical depth
3. Novel graph construction
4. Publishable at top venue

### For Maximum HPDC Impact

**Recommended: Combined Approach**

Core: IOGraphNet for detection
Add-on: LLM explanations for interpretability

This gives you:
- Technical novelty (GNN)
- Practical impact (explanations)
- Broader appeal to reviewers

---

## Part 11: Key References

### Must-Cite Papers

1. **AIIO** (HPDC'23) - Direct comparison baseline
   - Dong et al., "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis"

2. **Modular Darshan** (ESPT'16) - Data source foundation
   - Snyder et al., "Modular HPC I/O Characterization with Darshan"

3. **UMAMI** (PDSW-DISCS'17) - Holistic analysis concept
   - Lockwood et al., "UMAMI: A Recipe for Generating Meaningful Metrics"

4. **Zoom-in Analysis** (CCGrid'19) - Root cause analysis
   - Byna et al., "A Zoom-in Analysis of I/O Logs to Detect Root Causes"

5. **GNN-RL** (SC'24) - GNN in HPC context
   - Adimora et al., "GNN-RL: An Intelligent HPC Resource Scheduler"

6. **I/O Burst Prediction** (2023) - ML for Darshan
   - Saeedizade et al., "I/O Burst Prediction for HPC Clusters using Darshan Logs"

### Additional Important References

- DXT Explorer for visualization concepts
- Recorder for multi-layer tracing
- TOKIO for holistic analysis framework
- Root causes of cross-application I/O interference

---

## Part 12: Checklist for HPDC Submission

### Pre-Submission Checklist

- [ ] Paper ≤ 11 pages (excluding references)
- [ ] ACM Proceedings format
- [ ] Dual-anonymous compliant
- [ ] No identifying information
- [ ] Self-citations in third person
- [ ] AI tool usage disclosed (if applicable)
- [ ] Conflict of interest declared
- [ ] Supplementary materials prepared

### Technical Checklist

- [ ] Clear problem statement with motivation
- [ ] Quantitative gap analysis
- [ ] Novel technical contribution
- [ ] Rigorous experimental methodology
- [ ] Comparison with strong baselines
- [ ] Ablation studies
- [ ] Generalization experiments
- [ ] Limitations acknowledged
- [ ] Reproducibility information

### Presentation Checklist

- [ ] Clear figures (architecture, results)
- [ ] Informative tables
- [ ] Algorithm pseudocode
- [ ] Well-organized sections
- [ ] Proofread for clarity
- [ ] Native speaker review (if possible)

---

## Conclusion

This roadmap provides a comprehensive plan for developing novel AI/GNN approaches for I/O performance bottleneck detection targeting HPDC 2026. The key differentiators from existing work are:

1. **GNN-based modeling** - First of its kind for I/O bottlenecks
2. **Multi-module analysis** - Beyond POSIX-only approaches
3. **Inter-job contention** - Temporal graph modeling
4. **Interpretability** - LLM-augmented explanations
5. **Large-scale validation** - 1.37M production logs

With careful execution of this plan, you have a strong chance of acceptance at HPDC 2026.

---

## Appendix A: Detailed Feature List

### POSIX Module Features (70+ counters)

```
# Operation counts
POSIX_OPENS, POSIX_READS, POSIX_WRITES, POSIX_SEEKS, POSIX_STATS
POSIX_FSYNCS, POSIX_FDSYNCS, POSIX_FILENOS, POSIX_DUPS, POSIX_MMAPS

# Data volume
POSIX_BYTES_READ, POSIX_BYTES_WRITTEN
POSIX_MAX_BYTE_READ, POSIX_MAX_BYTE_WRITTEN

# Access patterns
POSIX_CONSEC_READS, POSIX_CONSEC_WRITES
POSIX_SEQ_READS, POSIX_SEQ_WRITES
POSIX_RW_SWITCHES
POSIX_STRIDE[1-4]_STRIDE, POSIX_STRIDE[1-4]_COUNT
POSIX_ACCESS[1-4]_ACCESS, POSIX_ACCESS[1-4]_COUNT

# Size histograms (10 buckets each for read/write)
POSIX_SIZE_READ_0_100 ... POSIX_SIZE_READ_1G_PLUS
POSIX_SIZE_WRITE_0_100 ... POSIX_SIZE_WRITE_1G_PLUS

# Alignment
POSIX_FILE_NOT_ALIGNED, POSIX_FILE_ALIGNMENT
POSIX_MEM_NOT_ALIGNED, POSIX_MEM_ALIGNMENT

# Timing
POSIX_F_OPEN_START_TIMESTAMP, POSIX_F_OPEN_END_TIMESTAMP
POSIX_F_READ_START_TIMESTAMP, POSIX_F_READ_END_TIMESTAMP
POSIX_F_WRITE_START_TIMESTAMP, POSIX_F_WRITE_END_TIMESTAMP
POSIX_F_CLOSE_START_TIMESTAMP, POSIX_F_CLOSE_END_TIMESTAMP
POSIX_F_READ_TIME, POSIX_F_WRITE_TIME, POSIX_F_META_TIME

# Parallelism (shared files)
POSIX_FASTEST_RANK, POSIX_FASTEST_RANK_BYTES
POSIX_SLOWEST_RANK, POSIX_SLOWEST_RANK_BYTES
POSIX_F_FASTEST_RANK_TIME, POSIX_F_SLOWEST_RANK_TIME
POSIX_F_VARIANCE_RANK_TIME, POSIX_F_VARIANCE_RANK_BYTES
```

### MPI-IO Module Features (30+ counters)

```
# Operation counts
MPIIO_INDEP_OPENS, MPIIO_COLL_OPENS
MPIIO_INDEP_READS, MPIIO_INDEP_WRITES
MPIIO_COLL_READS, MPIIO_COLL_WRITES
MPIIO_SPLIT_READS, MPIIO_SPLIT_WRITES
MPIIO_NB_READS, MPIIO_NB_WRITES
MPIIO_SYNCS, MPIIO_HINTS, MPIIO_VIEWS

# Data volume
MPIIO_BYTES_READ, MPIIO_BYTES_WRITTEN

# Timing
MPIIO_F_READ_TIME, MPIIO_F_WRITE_TIME, MPIIO_F_META_TIME
```

### Derived Features (Computed)

```python
# Efficiency metrics
avg_read_size = BYTES_READ / READS
avg_write_size = BYTES_WRITTEN / WRITES
read_bandwidth = BYTES_READ / runtime
write_bandwidth = BYTES_WRITTEN / runtime

# Ratios
metadata_ratio = META_TIME / (READ_TIME + WRITE_TIME + META_TIME)
seq_ratio = (SEQ_READS + CONSEC_READS) / READS
alignment_eff = 1 - (FILE_NOT_ALIGNED / total_ops)
mpiio_posix_ratio = MPIIO_OPS / POSIX_OPS
collective_ratio = COLL_WRITES / (COLL_WRITES + INDEP_WRITES)

# Imbalance
imbalance_factor = SLOWEST_RANK_TIME / FASTEST_RANK_TIME
```

---

## Appendix B: Tool and Library Recommendations

### Data Processing
- PyDarshan: Darshan log parsing
- Pandas: Feature engineering
- PyArrow/Parquet: Efficient storage

### GNN Implementation
- PyTorch Geometric (PyG): GNN framework
- DGL: Alternative GNN library
- NetworkX: Graph analysis

### LLM Integration
- Hugging Face Transformers
- LangChain: RAG implementation
- OpenAI API / Local LLM (Llama)

### Experiment Tracking
- Weights & Biases (wandb)
- MLflow
- TensorBoard

### Visualization
- Matplotlib/Seaborn: Plots
- Plotly: Interactive visualization
- GraphViz: Graph visualization
