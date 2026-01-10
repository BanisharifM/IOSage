# HPDC 2026 Practical Roadmap: GNN-Based I/O Bottleneck Detection

## Executive Summary

This is a **practical, actionable roadmap** for developing a GNN-based I/O bottleneck detection system targeting HPDC 2026. Based on comprehensive research of existing tools, papers, and evaluation methodologies, this document provides specific steps to maximize your chances of acceptance.

**Core Approach:** GNN for multi-module I/O bottleneck detection
**Optional Extension:** LLM for interpretable explanations (if time permits)
**Demo:** Streamlit web application for visualization

---

## Part 1: Competitive Landscape

### 1.1 Existing Tools (Must Compare Against)

| Tool | Type | Strengths | Weaknesses | Source |
|------|------|-----------|------------|--------|
| **AIIO** | ML (XGBoost) | HPDC'23, job-level detection, SHAP | POSIX+Lustre only, no graph | [GitHub](https://github.com/hpc-io/aiio) |
| **Drishti** | Rule-based | Actionable recommendations, multi-module | No ML, threshold-based | [GitHub](https://github.com/hpc-io/drishti-io) |
| **DXT Explorer** | Visualization | Interactive, detailed traces | Manual analysis, no automation | [GitHub](https://github.com/hpc-io/dxt-explorer) |
| **ION** | LLM (HotStorage'24) | In-context learning, interactive | No structured detection, GPT-dependent | [Paper](https://dl.acm.org/doi/10.1145/3655038.3665950) |
| **IOAgent** | LLM (IPDPS'25) | RAG, TraceBench dataset, multi-LLM | Recent competitor, sets high bar | [Paper](https://ieeexplore.ieee.org/document/11078545/) |

### 1.2 Your Novelty vs Competitors

| Aspect | AIIO | Drishti | ION | IOAgent | **Your GNN** |
|--------|------|---------|-----|---------|--------------|
| Multi-module (POSIX+MPI-IO+STDIO) | Partial | Yes | Yes | Yes | **Yes** |
| Beyond Lustre (e.g., GPU I/O) | No | Limited | No | No | **Yes** |
| Graph structure (job relations) | No | No | No | No | **Yes (Novel)** |
| Inter-job contention modeling | No | No | No | No | **Yes (Novel)** |
| AI workload awareness | No | No | No | No | **Yes** |
| Scalability to 1M+ logs | Unknown | Yes | No | Unknown | **Yes** |

### 1.3 Key Papers to Cite

**Must-Cite (Direct Competitors):**
1. AIIO (HPDC'23) - Dong et al. - ML baseline
2. IOAgent (IPDPS'25) - Egersdoerfer et al. - LLM baseline
3. ION (HotStorage'24) - Egersdoerfer et al. - LLM approach
4. Drishti (ISC'23) - Bez et al. - Rule-based baseline

**Supporting Citations:**
5. Modular Darshan (ESPT'16) - Data source
6. DXT Explorer (PDSW'21) - Visualization approach
7. HPCGCN (2023) - GNN in HPC (not I/O)
8. GNN-RL (SC'24) - GNN for HPC scheduling

---

## Part 2: Your Dataset Advantage

### 2.1 ALCF Polaris Darshan Logs

| Attribute | Value | Significance |
|-----------|-------|--------------|
| Total Logs | 1,378,117 | Largest public Darshan dataset |
| Time Span | Apr 2024 - Jan 2026 | Recent, reflects modern workloads |
| System | Polaris (ALCF) | A100 GPUs, AI/ML heavy |
| Modules | POSIX, MPI-IO, STDIO, APMPI, HEATMAP | Multi-module analysis possible |
| Storage | Eagle/Grand Lustre (100 PiB) | Large-scale parallel I/O |

### 2.2 Why Polaris Data is Unique

1. **AI/ML Workloads**: 4x A100 GPUs per node = heavy deep learning I/O
2. **Modern Patterns**: 2024-2026 data captures current scientific computing trends
3. **Scale**: 1.37M logs provides statistical power for ML
4. **Multi-Module**: Beyond POSIX-only analysis

### 2.3 Estimated Workload Distribution (To Verify)

Based on Polaris usage patterns:
- ~40-50% AI/ML training workloads (PyTorch, TensorFlow)
- ~30-40% Traditional HPC simulations
- ~10-20% Data analytics and visualization

**Action Item:** Profile your dataset to confirm workload distribution.

---

## Part 3: Ground Truth Strategy

### 3.0 Recommended Approach: COMBINED

**Use BOTH benchmarks AND real data, but for different purposes:**

| Data Source | Size | Labels | Purpose |
|-------------|------|--------|---------|
| **Benchmarks** | ~3,000 logs | Definitive (you created the pattern) | Train/Test, report main metrics |
| **Polaris (real)** | 1.37M logs | None or weak (rule-based) | Generalization, clustering, scale demo |

**Why Combined is Strongest for HPDC:**
- Benchmarks only → Reviewers ask "Does it work on real data?"
- Real data only → Reviewers ask "How do you verify labels?" (circular reasoning)
- **Combined** → Both questions answered

**In Your Paper:**
1. Report Precision/Recall/F1 on **benchmark test set** (definitive ground truth)
2. Show generalization to **Polaris data** (clustering quality, agreement with Drishti)

### 3.1 How Many Ground Truth Samples Do You Need?

Based on ML literature research:

| Scenario | Samples Per Label | Total Benchmark Logs | Source |
|----------|-------------------|----------------------|--------|
| **Minimum viable** | 50-75 | 1,000-1,500 | [BMC Medical Informatics](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8) |
| **Recommended** | 100-200 | 2,000-3,000 | Statistical power analysis |
| **Robust** | 300+ | 4,000+ | Deep learning best practices |

**Note:** For multi-label classification, you need enough samples where each label appears. Some labels may co-occur (e.g., small_io + random_access), which is fine.

**Recommendation:** Aim for **2,000-3,000 benchmark logs** with multi-label annotations.

#### Label Distribution Target

| Label | Target Samples | Benchmark Source |
|-------|----------------|------------------|
| Small I/O | 200+ | IOR small transfer |
| Random Access | 200+ | IOR random |
| Misaligned | 150+ | IOR unaligned |
| Metadata Heavy | 200+ | mdtest |
| Excessive Opens | 150+ | Many-files benchmark |
| Load Imbalance | 150+ | Custom script |
| Missing Collective | 200+ | IOR POSIX mode |
| Stragglers | 100+ | Delayed rank script |
| Contention | 100+ | Concurrent jobs |
| OST Imbalance | 100+ | Specific stripe config |
| Checkpoint Bottleneck | 200+ | DLIO checkpoint |
| Data Loading Slow | 200+ | DLIO data loading |
| No Bottleneck | 300+ | Optimized benchmarks |

### 3.2 Ground Truth Sources

#### Source 1: Controlled I/O Benchmarks (Primary)

Run benchmarks with **intentional bottleneck patterns**:

| Benchmark | Bottleneck Type | I/O Pattern | Multi-Module? | AI Workload? |
|-----------|-----------------|-------------|---------------|--------------|
| **IOR** | B1: Small I/O, B5: Misaligned, B6: Random | Synthetic, configurable | POSIX, MPI-IO | No |
| **mdtest** | B2: Metadata | Metadata-heavy | POSIX | No |
| **IO500** | Comprehensive | Bandwidth + IOPS | POSIX, MPI-IO | No |
| **h5bench** | HDF5 patterns | Structured data | POSIX, HDF5 | No |
| **DLIO** | ML I/O patterns | Checkpoint, data loading | POSIX, TFRecord, HDF5 | **Yes** |
| **E3SM-IO** | Scientific patterns | Non-contiguous, small I/O | PnetCDF, MPI-IO | No |

#### Source 2: Drishti Rule-Based Labels (Secondary)

Use Drishti to auto-label Polaris logs:

```bash
# Run Drishti on all logs
drishti path/to/log.darshan --issues --recommendations

# Parse output for bottleneck labels
# Use as weak supervision for pre-training
```

#### Source 3: TraceBench Dataset (IOAgent)

The IOAgent paper mentions **TraceBench** - a labeled I/O trace test suite. Contact authors or check if released.

### 3.3 Recommended Ground Truth Collection Plan (Multi-Label)

```
Phase 1: Benchmark Experiments (Target: 3,000 samples)
├── IOR experiments: 800 logs
│   ├── Small I/O patterns: 200 logs
│   ├── Random access patterns: 200 logs
│   ├── Misaligned patterns: 150 logs
│   ├── Missing collective (POSIX mode): 150 logs
│   └── Optimized (no bottleneck): 100 logs
├── mdtest experiments: 300 logs
│   ├── Metadata heavy: 200 logs
│   └── Excessive opens: 100 logs
├── DLIO experiments: 600 logs (AI workloads - critical!)
│   ├── Checkpoint bottleneck: 200 logs
│   ├── Data loading slow: 200 logs
│   └── Mixed patterns: 200 logs
├── h5bench experiments: 300 logs (HDF5 patterns)
├── IO500 experiments: 300 logs (comprehensive)
├── Custom scripts: 400 logs
│   ├── Load imbalance: 150 logs
│   ├── Stragglers: 100 logs
│   └── Contention (concurrent jobs): 150 logs
└── Optimized benchmarks: 300 logs (no_bottleneck class)

Phase 2: Multi-Label Annotation
├── Each log gets multiple labels (not mutually exclusive)
├── Use detection rules to auto-annotate
├── Manual review of edge cases
└── Store as binary label vectors

Phase 3: Drishti Cross-Validation (Weak Supervision)
├── Run Drishti on all benchmark logs
├── Compare Drishti issues with your labels
├── Use disagreements to refine labeling
└── Run on 50K Polaris logs for pre-training

Phase 4: Quality Control
├── Sample 300 logs for manual verification
├── Compute inter-annotator agreement (if multiple labelers)
├── Document labeling criteria clearly
└── Release labeling guidelines with code
```

#### Multi-Label Annotation Example

```python
# Example: A job with BOTH small I/O AND random access
log_file = "job_12345.darshan"
labels = [
    1,  # small_io: YES (avg transfer < 1KB)
    1,  # random_access: YES (seq_ratio < 0.3)
    0,  # misaligned: NO
    0,  # metadata_heavy: NO
    0,  # excessive_opens: NO
    0,  # load_imbalance: NO
    0,  # missing_collective: NO
    0,  # stragglers: NO
    0,  # contention: NO
    0,  # ost_imbalance: NO
    0,  # checkpoint_bottleneck: NO
    0,  # data_loading_slow: NO
    0,  # no_bottleneck: NO (has issues!)
]
```

---

## Part 4: Benchmarks for Different I/O Patterns

### 4.1 Traditional HPC Benchmarks

| Benchmark | Pattern | Command Example | Bottleneck Created |
|-----------|---------|-----------------|-------------------|
| **IOR (small)** | Small I/O | `mpirun -np 64 ior -t 64 -b 64K` | B1: Small I/O |
| **IOR (random)** | Random access | `mpirun -np 64 ior -z -t 1M` | B6: Random access |
| **IOR (unaligned)** | Misaligned | `mpirun -np 64 ior -t 1000` | B5: Misalignment |
| **IOR (independent)** | No collective | `mpirun -np 64 ior -a POSIX` | B4: Missing collective |
| **mdtest** | Metadata | `mpirun -np 64 mdtest -n 10000` | B2: Metadata heavy |
| **IO500** | Mixed | `io500.sh config.ini` | Comprehensive |

### 4.2 AI/ML Workload Benchmarks (Critical for Polaris Data)

| Benchmark | Workload | Pattern | Command |
|-----------|----------|---------|---------|
| **DLIO (UNet3D)** | Medical imaging | Large checkpoint I/O | `dlio_benchmark workload=unet3d` |
| **DLIO (ResNet50)** | Image classification | Small file reads | `dlio_benchmark workload=resnet50` |
| **DLIO (BERT)** | NLP | Text data loading | `dlio_benchmark workload=bert` |
| **h5bench** | HDF5 checkpoint | Structured arrays | `h5bench config.json` |

### 4.3 Scientific Application I/O Kernels

| Kernel | Domain | Pattern |
|--------|--------|---------|
| **E3SM-IO** | Climate | Non-contiguous, many small writes |
| **HACC-IO** | Cosmology | Large particle data |
| **VPIC-IO** | Physics | Particle checkpointing |
| **MADBench2** | CMB analysis | Dense matrix I/O |

---

## Part 5: Model Architecture

### 5.1 GNN Design for I/O Bottleneck Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    IOGraphNet Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Darshan Log                                             │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Multi-Module Feature Extraction                        │   │
│  │  ├── POSIX counters (70+ features)                      │   │
│  │  ├── MPI-IO counters (30+ features)                     │   │
│  │  ├── STDIO counters (20+ features)                      │   │
│  │  └── Derived metrics (bandwidth, ratios, etc.)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Graph Construction                                      │   │
│  │  ├── Nodes: File records from each module               │   │
│  │  ├── Edges: Same-file (cross-module), temporal          │   │
│  │  └── Job-level edges: Temporal overlap, resource sharing│   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Heterogeneous GNN Layers                               │   │
│  │  ├── GraphSAGE / GAT for message passing                │   │
│  │  ├── Module-specific encoders                           │   │
│  │  └── Cross-module attention                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output Heads                                            │   │
│  │  ├── Bottleneck classification (7 types)                │   │
│  │  ├── Severity scoring (0-1)                             │   │
│  │  └── Performance prediction (bandwidth)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  OUTPUT: Bottleneck type + Severity + Predicted bandwidth      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Bottleneck Types (12 Categories - Multi-Label)

**Why Multi-Label?** Real jobs often have MULTIPLE bottlenecks simultaneously (e.g., small I/O + random access). Single-label classification is unrealistic.

#### Category 1: Data Transfer Issues

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B1 | Small I/O | SIZE_0_100 + SIZE_100_1K > 80% ops | IOR `-t 64 -b 64K` |
| B2 | Random Access | SEQ_READS/READS < 0.3 | IOR `-z` flag |
| B3 | Misaligned Access | FILE_NOT_ALIGNED > 50% ops | IOR `-t 1000` |

#### Category 2: Metadata Issues

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B4 | Metadata Dominance | META_TIME > READ_TIME + WRITE_TIME | mdtest |
| B5 | Excessive Opens | OPENS/runtime > threshold | Many small files |

#### Category 3: Parallelism Issues

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B6 | Load Imbalance | VARIANCE_RANK_TIME > 0.5 | Custom imbalanced script |
| B7 | Missing Collective | MPIIO_INDEP > 0, MPIIO_COLL == 0 | IOR POSIX vs MPI-IO |
| B8 | Stragglers | SLOWEST_RANK_TIME / FASTEST_RANK_TIME > 2x | Delayed rank script |

#### Category 4: System Issues

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B9 | Contention | Low BW during high system load | Concurrent jobs |
| B10 | OST Imbalance | Uneven Lustre OST usage | Specific stripe config |

#### Category 5: AI/ML Specific (Important for Polaris Data)

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B11 | Checkpoint Bottleneck | Large periodic writes, high write time | DLIO checkpoint |
| B12 | Data Loading Slow | Many small reads, low read BW | DLIO data loading |

#### Category 6: Healthy

| ID | Type | Detection Criteria | Benchmark |
|----|------|--------------------|-----------------------|
| B0 | No Bottleneck | High BW, balanced, efficient patterns | Optimized IOR |

#### Multi-Label Output Format

```python
# Each job has a binary vector (multiple 1s allowed)
labels = {
    'small_io': 1,           # Present
    'random_access': 1,      # Present (co-occurring!)
    'misaligned': 0,
    'metadata_heavy': 0,
    'excessive_opens': 0,
    'load_imbalance': 0,
    'missing_collective': 0,
    'stragglers': 0,
    'contention': 0,
    'ost_imbalance': 0,
    'checkpoint_bottleneck': 0,
    'data_loading_slow': 0,
    'no_bottleneck': 0,
}
```

### 5.3 Why GNN Over Alternatives

| Approach | Multi-Module | Job Relations | Multi-Label | Interpretable | Scalable |
|----------|-------------|---------------|-------------|---------------|----------|
| Rule-based (Drishti) | Yes | No | Yes | Yes | Yes |
| XGBoost (AIIO) | Partial | No | No* | SHAP | Yes |
| MLP | Yes | No | Yes | Limited | Yes |
| **GNN (Ours)** | **Yes** | **Yes** | **Yes** | **Attention** | **Yes** |
| LLM (IOAgent) | Yes | No | Yes | Yes | Limited |

*AIIO uses regression, not classification

### 5.4 Multi-Task GNN Architecture

**Why Multi-Task?** Classification alone is not enough. We need:
- Which bottlenecks? (classification)
- How bad is performance? (regression)
- How severe is each bottleneck? (severity scoring)
- Why? (interpretation via attention)

```python
class MultiTaskIOGraphNet(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_labels=13):
        super().__init__()

        # Shared GNN Encoder (GAT for attention-based interpretation)
        self.conv1 = GATConv(in_features, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1)

        # Task 1: Multi-label bottleneck classification
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Task 2: Bandwidth/performance regression
        self.regressor = nn.Linear(hidden_dim, 1)

        # Task 3: Severity scoring (per bottleneck type)
        self.severity = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, edge_index, batch):
        # GNN encoding with attention weights
        h, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        h = F.elu(h)
        h, attn2 = self.conv2(h, edge_index, return_attention_weights=True)
        h = F.elu(h)
        h, attn3 = self.conv3(h, edge_index, return_attention_weights=True)

        # Global pooling (graph-level representation)
        h_graph = global_mean_pool(h, batch)

        # Multi-task outputs
        bottleneck_logits = self.classifier(h_graph)
        bandwidth = self.regressor(h_graph)
        severity = torch.sigmoid(self.severity(h_graph))

        return {
            'bottlenecks': torch.sigmoid(bottleneck_logits),  # 13 probs
            'bandwidth': bandwidth,                             # MB/s
            'severity': severity,                               # 13 scores
            'attention': (attn1, attn2, attn3)                 # Interpretation
        }

# Multi-task loss
def multi_task_loss(pred, target, alpha=1.0, beta=0.5, gamma=0.5):
    # Task 1: Classification loss
    cls_loss = F.binary_cross_entropy(pred['bottlenecks'], target['labels'])

    # Task 2: Regression loss (bandwidth)
    reg_loss = F.mse_loss(pred['bandwidth'], target['bandwidth'])

    # Task 3: Severity loss
    sev_loss = F.mse_loss(pred['severity'], target['severity'])

    return alpha * cls_loss + beta * reg_loss + gamma * sev_loss
```

### 5.5 GNN Architectures to Compare

| Architecture | Description | Interpretability |
|--------------|-------------|------------------|
| **GAT** | Graph Attention Network | **Attention weights** (recommended) |
| **GraphSAGE** | Sampling + Aggregation | Feature importance |
| **GIN** | Graph Isomorphism Network | Limited |
| **HeteroGNN** | Heterogeneous (different node types) | Per-type attention |

**Recommendation:** Start with **GAT** because attention weights provide built-in interpretability.

### 5.6 Interpretation Pipeline

```
INPUT: Darshan Log
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ GNN PREDICTION                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Bottlenecks: [small_io=0.92, load_imbalance=0.78, ...]         │
│ Bandwidth: 45 MB/s (expected: 650 GB/s)                        │
│ Severity: small_io=HIGH, load_imbalance=MEDIUM                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ INTERPRETATION (Why?)                                           │
├─────────────────────────────────────────────────────────────────┤
│ Attention weights → File A (0.8), File B (0.6) are problematic │
│ Feature importance → SIZE_READ_0_100 is high (85% small reads) │
│ Cross-module → MPI-IO to POSIX ratio is 1:1 (no aggregation)  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RECOMMENDATION (How to fix?)                                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Increase buffer size to 1MB+ (fixes small_io)               │
│ 2. Use MPI_File_write_all (fixes load_imbalance)               │
│ 3. Expected improvement: 10-50x                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.7 Recommendation Mapping

| Bottleneck | Root Cause | Recommendation |
|------------|------------|----------------|
| Small I/O | Transfer size <1KB | Buffer writes, use 1MB+ transfers |
| Random Access | Non-sequential offsets | Sort I/O, use collective |
| Misaligned | Not aligned to stripe | Align to 1MB (Lustre stripe) |
| Metadata Heavy | Too many files/ops | Reduce file count, cache handles |
| Excessive Opens | Open/close per write | Keep files open, batch operations |
| Load Imbalance | Uneven rank distribution | Collective I/O, redistribute |
| Missing Collective | Independent I/O | Use MPI_File_write_all |
| Stragglers | One rank slow | Check contention, redistribute |
| Contention | Shared resources | Stagger I/O, use local storage |
| OST Imbalance | Single OST | Increase stripe count |
| Checkpoint Bottleneck | Large periodic writes | Async I/O, compression |
| Data Loading Slow | Small file reads | Prefetch, parallel loaders |

---

## Part 6: Evaluation Strategy

### 6.1 Multi-Layer Validation (No Single Point of Failure)

```
Layer 1: CONTROLLED BENCHMARKS
├── IOR, mdtest, DLIO, h5bench with known patterns
├── Precision/Recall/F1 per bottleneck type
└── This is your PRIMARY evaluation

Layer 2: CROSS-TOOL AGREEMENT
├── Compare with Drishti predictions
├── Compare with AIIO predictions (if possible)
└── High agreement = confidence

Layer 3: ABLATION STUDIES
├── GNN vs MLP (prove graph structure helps)
├── Multi-module vs POSIX-only
├── With/without inter-job edges
└── Feature importance analysis

Layer 4: CLUSTERING QUALITY (No Ground Truth Needed)
├── Silhouette score on Polaris logs
├── Cluster visualization
└── Interpretable clusters?

Layer 5: LIMITED INTERVENTION (5-10 benchmarks)
├── Run benchmark → detect bottleneck
├── Apply recommendation → rerun
└── Measure improvement
```

### 6.2 Specific Metrics to Report (Multi-Label)

#### Multi-Label Classification Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hamming Loss** | Fraction of wrong labels (lower is better) | < 0.15 |
| **Subset Accuracy** | Exact match of all labels | > 60% |
| **Micro F1** | F1 across all label predictions | > 0.80 |
| **Macro F1** | Average F1 per label | > 0.75 |
| **Jaccard Score** | Intersection over union of labels | > 0.70 |

#### Per-Label Metrics

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Small I/O | > 0.80 | > 0.75 | > 0.77 | - |
| Random Access | > 0.75 | > 0.70 | > 0.72 | - |
| Metadata Heavy | > 0.80 | > 0.75 | > 0.77 | - |
| ... (all 13 labels) | | | | |

#### Multi-Label Evaluation Code

```python
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,      # subset accuracy for multi-label
    f1_score,
    jaccard_score,
    classification_report,
    multilabel_confusion_matrix
)

# Multi-label predictions (threshold at 0.5)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# Metrics
hamming = hamming_loss(y_true, y_pred_binary)
subset_acc = accuracy_score(y_true, y_pred_binary)  # Exact match
micro_f1 = f1_score(y_true, y_pred_binary, average='micro')
macro_f1 = f1_score(y_true, y_pred_binary, average='macro')
jaccard = jaccard_score(y_true, y_pred_binary, average='samples')

# Per-label report
print(classification_report(y_true, y_pred_binary, target_names=label_names))
```

#### Other Metrics

| Category | Metric | Target |
|----------|--------|--------|
| **Clustering** | Silhouette score | > 0.5 |
| **Ablation** | GNN vs MLP improvement | > 5% |
| **Ablation** | Multi-module vs POSIX | > 3% |
| **Intervention** | Performance improvement | > 2x on avg |

### 6.3 Baselines to Compare Against

| Baseline | How to Obtain | Priority |
|----------|---------------|----------|
| **AIIO** | [GitHub](https://github.com/hpc-io/aiio) | Must |
| **Drishti** | [GitHub](https://github.com/hpc-io/drishti-io) | Must |
| **XGBoost** | Implement yourself | Must |
| **Random Forest** | Implement yourself | Must |
| **MLP** | Implement yourself | Must |
| **IOAgent** | Contact authors or implement | Optional |

---

## Part 7: LLM Extension (Optional)

### 7.1 When to Add LLM

Add LLM component **only if**:
1. GNN achieves target metrics (>85% accuracy)
2. You have time remaining before deadline
3. You can validate explanations

### 7.2 If Adding LLM

**Architecture:**
```
GNN Detection → Bottleneck + Key Metrics → LLM Prompt → Explanation
```

**Validation Methods:**
1. Factual consistency checking (claims vs metrics)
2. Self-consistency (same input → similar explanations)
3. LLM-as-judge (GPT-4 rates explanation quality)

**Comparison:** Must compare against ION and IOAgent if including LLM.

---

## Part 8: Streamlit Demo Application

### 8.1 Purpose

- Provides visual evidence for paper
- Makes tool accessible to non-experts
- Can include in supplementary materials

### 8.2 Recommended Features

```
┌─────────────────────────────────────────────────────────────────┐
│                    IOGraphNet Web Demo                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Upload Darshan Log]  or  [Paste Job ID]                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  JOB SUMMARY                                             │   │
│  │  Job ID: 12345    Runtime: 23.5 min    Ranks: 64        │   │
│  │  Total Read: 37.6 GiB    Total Write: 1.0 GiB           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DETECTED BOTTLENECKS                                    │   │
│  │  🔴 Small I/O (Severity: 0.87)                          │   │
│  │  🟡 Metadata Overhead (Severity: 0.45)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RECOMMENDATIONS                                         │   │
│  │  1. Increase transfer size to 1MB+                      │   │
│  │  2. Use collective MPI-IO operations                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GRAPH VISUALIZATION                                     │   │
│  │  [Interactive graph showing file/module relationships]  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Implementation Stack

```python
# requirements.txt
streamlit>=1.28
torch>=2.0
torch-geometric>=2.4
darshan>=3.4
plotly>=5.18
networkx>=3.2
```

---

## Part 9: Implementation Phases

### Phase 1: Data Preparation

**Tasks:**
1. Parse all 1.37M Darshan logs with PyDarshan
2. Extract multi-module features (POSIX + MPI-IO + STDIO)
3. Store in efficient format (Parquet/HDF5)
4. Run Drishti on subset for weak labels

**Deliverables:**
- Feature database
- Basic statistics on dataset
- Workload distribution analysis

### Phase 2: Benchmark Ground Truth Collection

**Tasks:**
1. Set up benchmark environment (IOR, mdtest, DLIO, h5bench)
2. Run benchmarks with each bottleneck pattern
3. Collect Darshan logs
4. Label with known ground truth

**Deliverables:**
- 2,000+ labeled benchmark logs
- Per-bottleneck-type samples

### Phase 3: Model Development

**Tasks:**
1. Implement graph construction pipeline
2. Implement GNN architecture (PyTorch Geometric)
3. Train on benchmark ground truth
4. Validate on held-out benchmarks

**Deliverables:**
- Trained GNN model
- Training curves
- Initial accuracy metrics

### Phase 4: Evaluation

**Tasks:**
1. Full ablation studies
2. Comparison with baselines (AIIO, Drishti, XGBoost)
3. Clustering analysis on Polaris logs
4. Limited intervention study

**Deliverables:**
- Complete evaluation results
- Comparison tables
- Ablation analysis

### Phase 5: Demo & Paper

**Tasks:**
1. Build Streamlit application
2. Write paper (11 pages)
3. Prepare figures and tables
4. Internal review

**Deliverables:**
- Working demo
- Paper draft
- Supplementary materials

### Phase 6: LLM Extension (Optional)

**Tasks:**
1. Integrate LLM for explanations
2. Validate explanations
3. Compare with IOAgent

**Deliverables:**
- LLM integration
- Explanation quality metrics

---

## Part 10: Tools and Libraries

### Data Processing
| Tool | Purpose | Link |
|------|---------|------|
| PyDarshan | Parse Darshan logs | [Docs](https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/) |
| darshan-util | C tools for parsing | [Docs](https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html) |
| Pandas | Feature engineering | Built-in |
| PyArrow | Efficient storage | [Docs](https://arrow.apache.org/docs/python/) |

### GNN Implementation
| Tool | Purpose | Link |
|------|---------|------|
| PyTorch Geometric | GNN framework | [Docs](https://pytorch-geometric.readthedocs.io/) |
| NetworkX | Graph analysis | [Docs](https://networkx.org/) |
| DGL | Alternative GNN | [Docs](https://www.dgl.ai/) |

### Benchmarks
| Tool | Purpose | Link |
|------|---------|------|
| IOR | Synthetic I/O | [GitHub](https://github.com/hpc/ior) |
| mdtest | Metadata benchmark | [GitHub](https://github.com/hpc/ior) |
| DLIO | AI workload I/O | [GitHub](https://github.com/argonne-lcf/dlio_benchmark) |
| h5bench | HDF5 I/O | [GitHub](https://github.com/hpc-io/h5bench) |
| IO500 | Comprehensive | [Website](https://io500.org/) |

### Validation
| Tool | Purpose | Link |
|------|---------|------|
| Drishti | Rule-based comparison | [GitHub](https://github.com/hpc-io/drishti-io) |
| DXT Explorer | Visual validation | [GitHub](https://github.com/hpc-io/dxt-explorer) |
| scikit-learn | Silhouette score | Built-in |

### Demo
| Tool | Purpose | Link |
|------|---------|------|
| Streamlit | Web application | [Docs](https://streamlit.io/) |
| Plotly | Interactive plots | [Docs](https://plotly.com/python/) |

---

## Part 11: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GNN doesn't beat baselines | Medium | High | Start with strong baselines, have backup |
| Not enough benchmark data | Low | High | Start benchmark collection early |
| IOAgent outperforms | Medium | Medium | Focus on graph novelty, not LLM |
| Evaluation questioned | Medium | High | Multi-layer validation |
| Time runs out | Medium | High | Prioritize GNN, LLM is optional |

---

## Part 12: HPDC Submission Checklist

### Paper Requirements
- [ ] 11 pages max (excluding references)
- [ ] ACM Proceedings format
- [ ] Dual-anonymous (no author info)
- [ ] Paper type indicated in title

### Technical Content
- [ ] Clear novelty statement (first GNN for I/O bottleneck)
- [ ] Comparison with AIIO, Drishti
- [ ] Multi-layer evaluation
- [ ] Ablation studies
- [ ] Limitations section
- [ ] Reproducibility info

### Figures
- [ ] Architecture diagram
- [ ] Graph construction example
- [ ] Confusion matrix
- [ ] Ablation results
- [ ] Benchmark comparison
- [ ] Streamlit demo screenshot

---

## Part 13: HPC Relevance (Critical for HPDC)

HPDC explicitly focuses on **High-Performance Parallel and Distributed Computing**. Your paper must clearly demonstrate HPC relevance throughout.

### 13.1 HPC Aspects of Your Work

| Aspect | How Your Work is HPC-Relevant |
|--------|------------------------------|
| **Data Source** | 1.37M logs from ALCF Polaris, a DOE leadership-class supercomputer |
| **Scale** | 520 nodes, 2,080 GPUs, 100 PiB parallel file system |
| **I/O Stack** | Lustre (parallel FS), MPI-IO (parallel I/O library), HDF5 |
| **Workloads** | Scientific simulations, AI/ML training at scale |
| **Problem** | I/O bottleneck is critical HPC challenge (up to 90% of runtime) |
| **Impact** | Helps millions of node-hours be used more efficiently |

### 13.2 HPC-Specific Features in Your Approach

**Multi-Rank Analysis:**
- POSIX_VARIANCE_RANK_TIME - measures imbalance across MPI ranks
- FASTEST_RANK / SLOWEST_RANK - identifies stragglers
- Collective vs independent I/O - MPI-IO patterns

**Parallel File System Awareness:**
- Lustre OST (Object Storage Target) usage
- Stripe settings impact
- File alignment with stripe boundaries

**Large-Scale Patterns:**
- Checkpoint/restart I/O (common in HPC)
- N-to-1 vs N-to-N file patterns
- Shared file contention

### 13.3 HPC Keywords to Include in Paper

Use these terms throughout your paper:

```
- Supercomputer / Leadership-class computing
- Parallel file system (Lustre, GPFS)
- MPI-IO / Collective I/O
- Node-hours / Computational efficiency
- Production workloads
- Scientific applications
- Exascale computing
- I/O stack (application → library → file system)
- Parallel I/O performance
- HPC storage systems
```

### 13.4 HPC Impact Statement (For Paper)

> "I/O performance remains a critical bottleneck in HPC, often consuming up to 90% of application runtime for data-intensive workloads. On leadership-class systems like ALCF Polaris, even small improvements in I/O efficiency can save thousands of node-hours. Our GNN-based approach addresses this challenge by automatically detecting I/O bottlenecks across the complex HPC I/O stack (POSIX, MPI-IO, parallel file systems), enabling domain scientists to optimize their applications without deep I/O expertise."

### 13.5 HPC-Specific Evaluation Points

Include these in your evaluation:

1. **Scalability**: Show model works on 1M+ logs from real HPC system
2. **Multi-Rank**: Demonstrate detection of load imbalance across MPI ranks
3. **Parallel I/O**: Show collective vs independent detection
4. **Real Workloads**: Use actual scientific application patterns (E3SM-IO, HACC-IO)
5. **Production Environment**: All data from production HPC system

### 13.6 Why HPDC (Not Other Venues)

| Venue | Focus | Why HPDC is Better for You |
|-------|-------|---------------------------|
| SC | Broader HPC | Too competitive, HPDC more focused |
| IPDPS | Parallel/distributed | Good alternative, but HPDC has I/O track |
| CLUSTER | Cluster computing | Less prestigious |
| FAST | Storage | More file system focused |
| **HPDC** | HPC systems | Perfect fit for I/O + AI + HPC |

---

## Summary: What Makes This Paper Strong for HPDC

1. **Clear Novelty**: First GNN for I/O bottleneck detection
2. **HPC Focus**: Data from DOE leadership-class supercomputer
3. **Multi-Label**: Realistic detection of co-occurring bottlenecks (12 types)
4. **Multi-Module**: Beyond POSIX-only (addresses gap in AIIO)
5. **Large-Scale**: 1.37M production logs from real HPC system
6. **AI Workload Awareness**: Polaris A100 data (modern HPC)
7. **Parallel I/O Stack**: POSIX + MPI-IO + file system
8. **Rigorous Evaluation**: Multi-layer validation
9. **Practical Demo**: Streamlit application
10. **Strong Baselines**: AIIO, Drishti, IOAgent comparison
11. **HPC Impact**: Saves node-hours, helps scientists

---

## Quick Reference: Next Steps

1. **Now**: Set up benchmark environment (IOR, DLIO)
2. **Week 1-2**: Collect 2,000+ labeled benchmark logs
3. **Week 3-4**: Implement GNN, train on benchmarks
4. **Week 5-6**: Full evaluation, ablation studies
5. **Week 7-8**: Write paper, build demo
6. **Optional**: Add LLM if time permits

---

## References

### Papers
1. [AIIO (HPDC'23)](https://dl.acm.org/doi/10.1145/3588195.3592986)
2. [IOAgent (IPDPS'25)](https://ieeexplore.ieee.org/document/11078545/)
3. [ION (HotStorage'24)](https://dl.acm.org/doi/10.1145/3655038.3665950)
4. [Drishti (ISC'23)](https://jeanbez.gitlab.io/isc23/)
5. [HPCGCN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9893918/)

### Tools
1. [Drishti-IO](https://github.com/hpc-io/drishti-io)
2. [DXT Explorer](https://github.com/hpc-io/dxt-explorer)
3. [DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)
4. [h5bench](https://github.com/hpc-io/h5bench)
5. [IOR/mdtest](https://github.com/hpc/ior)

### Documentation
1. [HPDC 2025 CFP](https://hpdc.sci.utah.edu/2025/calls-cfp.html)
2. [PyDarshan](https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/)
3. [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
