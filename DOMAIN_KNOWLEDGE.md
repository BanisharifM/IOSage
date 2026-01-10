# Domain Knowledge: HPC I/O, Darshan, GNN, and ML

This document contains essential domain knowledge for the HPDC 2026 IOGraphNet project. **Read this before starting any implementation task.**

---

## 1. Darshan I/O Profiler

### 1.1 What is Darshan?

Darshan is a lightweight I/O characterization tool developed at Argonne National Laboratory that transparently captures I/O access patterns from HPC applications.

**Key Components:**
- **darshan-runtime**: Instruments MPI applications at runtime
- **darshan-util**: Command-line tools for parsing logs
- **PyDarshan**: Python API for log analysis (recommended)

**Documentation:**
- Official: https://darshan.readthedocs.io/
- NERSC: https://docs.nersc.gov/tools/performance/darshan/

### 1.2 Darshan Modules

| Module | What It Captures |
|--------|------------------|
| **POSIX** | Standard file operations (open, read, write, close, seek) |
| **MPI-IO** | Parallel I/O operations (collective, independent) |
| **STDIO** | Standard I/O library calls (fopen, fread, fprintf) |
| **HDF5** | Hierarchical Data Format operations |
| **PNetCDF** | Parallel NetCDF operations |
| **DXT** | Extended tracing (full I/O operation trace) |

### 1.3 Key Darshan Counters

**POSIX Module Counters:**
```
POSIX_READS          - Total read operations
POSIX_WRITES         - Total write operations
POSIX_OPENS          - File open count
POSIX_SEEKS          - Seek operations
POSIX_BYTES_READ     - Total bytes read
POSIX_BYTES_WRITTEN  - Total bytes written
POSIX_F_READ_TIME    - Time spent reading
POSIX_F_WRITE_TIME   - Time spent writing
POSIX_F_META_TIME    - Metadata operation time
POSIX_ACCESS1_COUNT  - Most common access size count
POSIX_ACCESS1_ACCESS - Most common access size value
POSIX_STRIDE1_*      - Access stride patterns
POSIX_SEQ_READS      - Sequential read count
POSIX_SEQ_WRITES     - Sequential write count
POSIX_CONSEC_READS   - Consecutive reads
POSIX_CONSEC_WRITES  - Consecutive writes
```

**MPI-IO Module Counters:**
```
MPIIO_INDEP_READS    - Independent read operations
MPIIO_INDEP_WRITES   - Independent write operations
MPIIO_COLL_READS     - Collective read operations
MPIIO_COLL_WRITES    - Collective write operations
MPIIO_SPLIT_READS    - Split collective reads
MPIIO_SPLIT_WRITES   - Split collective writes
MPIIO_HINTS          - MPI hints used
```

### 1.4 Parsing Darshan Logs

```bash
# Basic parsing
darshan-parser file.darshan

# Get job summary
darshan-job-summary.pl file.darshan

# Python (recommended)
import darshan
log = darshan.DarshanReport("file.darshan")
print(log.modules)  # Available modules
posix_df = log.records['POSIX'].to_df()  # POSIX as DataFrame
```

### 1.5 Important Limitation

Darshan only aggregates data when `MPI_Finalize()` is called. Crashed or incomplete jobs don't generate usable logs.

---

## 2. HPC I/O Benchmarks

### 2.1 IOR (Interleaved or Random)

**Purpose:** Standard benchmark for parallel I/O performance testing.

**Key Parameters:**
| Flag | Meaning |
|------|---------|
| `-a` | API (POSIX, MPIIO, HDF5) |
| `-b` | Block size per process |
| `-t` | Transfer size |
| `-s` | Segment count |
| `-F` | File-per-process mode |
| `-C` | Collective I/O |
| `-w` | Write test |
| `-r` | Read test |
| `-i` | Iterations |

**Example Commands:**
```bash
# Shared file with collective I/O
mpirun -n 64 ior -a MPIIO -b 1g -t 1m -w -r -C

# File-per-process with POSIX
mpirun -n 64 ior -a POSIX -b 1g -t 1m -w -r -F

# Small I/O (bottleneck pattern)
mpirun -n 64 ior -a POSIX -b 1m -t 4k -w -r
```

**What IOR Reveals:**
- Peak bandwidth (large transfers)
- Small I/O overhead
- Collective vs independent performance
- Metadata overhead with many files

**Documentation:** https://ior.readthedocs.io/

### 2.2 mdtest

**Purpose:** Metadata performance testing (creates/stats/removes many files).

**Key Parameters:**
| Flag | Meaning |
|------|---------|
| `-n` | Files per process |
| `-d` | Directory for test |
| `-i` | Iterations |
| `-u` | Unique directory per process |

**Example:**
```bash
mpirun -n 64 mdtest -n 1000 -d /scratch/test -i 3
```

### 2.3 DLIO (Deep Learning I/O)

**Purpose:** Emulates I/O patterns of deep learning training workloads.

**Key Features:**
- Replaces computation with sleep (isolates I/O)
- Supports multiple data formats (TFRecord, HDF5, NPZ, JPEG)
- Configurable batch size, prefetch, shuffle
- Achieves 90%+ similarity to real DL workloads

**Key Parameters:**
```yaml
# Example DLIO config
model: unet3d
framework: pytorch
workflow:
  generate_data: True
  train: True
dataset:
  data_folder: /data/train
  format: hdf5
  num_files_train: 1000
  record_length: 1048576  # 1MB samples
reader:
  batch_size: 4
  read_threads: 4
  prefetch_size: 2
```

**What DLIO Reveals:**
- Data loading bottlenecks
- Prefetch effectiveness
- Format efficiency (HDF5 vs TFRecord)
- Multi-threaded I/O scalability

**Documentation:** https://dlio-benchmark.readthedocs.io/

### 2.4 Other Benchmarks

| Benchmark | Purpose |
|-----------|---------|
| **IO500** | Comprehensive I/O benchmark suite (IOR + mdtest) |
| **h5bench** | HDF5-specific patterns |
| **E3SM-IO** | Climate simulation I/O patterns |
| **MACSio** | Multi-physics I/O patterns |

---

## 3. I/O Bottleneck Categories

### 3.1 The 13 Bottleneck Types

| ID | Category | Description | Detection Signal |
|----|----------|-------------|------------------|
| 1 | Small I/O | Transfers < 1MB | `ACCESS1_ACCESS < 1MB` |
| 2 | Misaligned I/O | Not aligned to stripe/block | Offset % stripe_size != 0 |
| 3 | Random Access | Non-sequential patterns | `SEQ_READS << READS` |
| 4 | Redundant Read | Same data read multiple times | Read bytes >> file size |
| 5 | Write Amplification | Excessive writes vs data size | Write bytes >> expected |
| 6 | Metadata Overhead | Too many open/stat/close | `META_TIME > 10% total` |
| 7 | Load Imbalance | Uneven I/O across ranks | Max/mean I/O ratio > 2 |
| 8 | Missing Collective | Independent when collective better | `COLL_* = 0`, many ranks |
| 9 | Lock Contention | Shared file with many writers | High variance in timing |
| 10 | Checkpoint Burst | Periodic large writes | Time series pattern |
| 11 | Data Loading Stall | GPU idle waiting for data | Compute/IO ratio |
| 12 | Poor Prefetch | Inefficient data pipeline | Sequential stalls |
| 13 | No Bottleneck | Well-optimized I/O | None of above |

### 3.2 Detection Heuristics

```python
def detect_small_io(posix_counters):
    """Small I/O if median access size < 1MB"""
    access_size = posix_counters['POSIX_ACCESS1_ACCESS']
    return access_size < 1024 * 1024

def detect_random_access(posix_counters):
    """Random if sequential reads < 50% of total reads"""
    seq = posix_counters['POSIX_SEQ_READS']
    total = posix_counters['POSIX_READS']
    return (seq / max(total, 1)) < 0.5

def detect_metadata_overhead(posix_counters):
    """Metadata overhead if meta_time > 10% of total"""
    meta = posix_counters['POSIX_F_META_TIME']
    total = meta + posix_counters['POSIX_F_READ_TIME'] + posix_counters['POSIX_F_WRITE_TIME']
    return (meta / max(total, 1e-9)) > 0.1
```

---

## 4. Graph Neural Networks (GNN)

### 4.1 Why GNN for I/O Analysis?

Traditional ML treats each job independently. GNN captures:
- **Temporal relationships**: Jobs that run simultaneously may compete for resources
- **File sharing**: Jobs accessing same files may have contention
- **User patterns**: Same user's jobs often have similar patterns
- **System state**: I/O performance depends on concurrent workload

### 4.2 Graph Construction

**Nodes:** Each job becomes a node with features from Darshan counters.

**Edges:** Connect jobs based on:
- **Temporal overlap**: Jobs running at same time
- **Shared files**: Jobs accessing same file paths
- **Same user**: Jobs from same user account
- **Same application**: Jobs running same executable

```python
# Example edge construction
def build_temporal_edges(jobs, window_seconds=300):
    edges = []
    for i, job_i in enumerate(jobs):
        for j, job_j in enumerate(jobs):
            if i < j:
                # Check temporal overlap
                overlap = min(job_i.end, job_j.end) - max(job_i.start, job_j.start)
                if overlap > 0:
                    edges.append((i, j))
    return edges
```

### 4.3 Node Features (from Darshan)

```python
# Feature vector per job
features = [
    # Volume
    log10(bytes_read + 1),
    log10(bytes_written + 1),

    # Operation counts
    log10(reads + 1),
    log10(writes + 1),

    # Access patterns
    seq_read_ratio,
    consec_read_ratio,

    # Access sizes (binned)
    small_io_fraction,  # < 4KB
    medium_io_fraction, # 4KB - 1MB
    large_io_fraction,  # > 1MB

    # Timing
    read_time_fraction,
    write_time_fraction,
    meta_time_fraction,

    # MPI-IO patterns
    collective_fraction,
    independent_fraction,

    # File patterns
    num_files,
    shared_file_flag,
]
```

### 4.4 GAT (Graph Attention Network)

**Why GAT?**
- Learns which neighbors are most relevant
- Attention weights are interpretable
- Works well with heterogeneous node types

**Architecture:**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class IOGraphNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # GNN layers with attention
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)

        # Multi-label classification
        return self.classifier(x)
```

### 4.5 Multi-Label Classification

For 13 bottleneck types, use BCEWithLogitsLoss:

```python
# Training
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    predictions = (torch.sigmoid(logits) > 0.5).float()
```

### 4.6 Metrics for Multi-Label

```python
from sklearn.metrics import f1_score, accuracy_score

# Micro-F1: treats all labels equally
micro_f1 = f1_score(y_true, y_pred, average='micro')

# Macro-F1: average F1 per label
macro_f1 = f1_score(y_true, y_pred, average='macro')

# Subset accuracy: exact match of all labels
subset_acc = accuracy_score(y_true, y_pred)

# Per-label F1
per_label_f1 = f1_score(y_true, y_pred, average=None)
```

---

## 5. PyTorch Geometric Essentials

### 5.1 Data Object

```python
from torch_geometric.data import Data

# Create a graph
data = Data(
    x=node_features,          # [num_nodes, num_features]
    edge_index=edge_index,    # [2, num_edges]
    edge_attr=edge_features,  # [num_edges, edge_features] (optional)
    y=labels,                 # [num_nodes, num_classes] for multi-label
)

# Useful properties
data.num_nodes
data.num_edges
data.num_node_features
data.has_self_loops()
data.is_directed()

# Move to GPU
data = data.to('cuda')
```

### 5.2 Edge Index Format

```python
# COO format: [2, num_edges]
# Row 0: source nodes
# Row 1: target nodes

edge_index = torch.tensor([
    [0, 1, 1, 2],  # source
    [1, 0, 2, 1],  # target
], dtype=torch.long)

# For undirected graphs, include both directions
```

### 5.3 DataLoader for Batching

```python
from torch_geometric.loader import DataLoader

# For graph-level tasks (multiple graphs)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# For node-level tasks (single large graph)
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # 2-hop neighbors
    batch_size=128,
    input_nodes=data.train_mask,
)
```

---

## 6. Self-Supervised Pre-training

### 6.1 Why Pre-train?

- Labeled data is expensive (need expert annotation)
- Unlabeled Darshan logs are abundant (1.37M)
- Pre-training learns general I/O patterns
- Fine-tuning adapts to specific bottleneck detection

### 6.2 Pre-training Tasks

**Contrastive Learning:**
```python
# Augment graph (drop edges, mask features)
# Learn embeddings where similar jobs are close

def augment_graph(data, edge_drop_prob=0.2, feat_mask_prob=0.1):
    # Drop edges randomly
    mask = torch.rand(data.edge_index.size(1)) > edge_drop_prob
    edge_index = data.edge_index[:, mask]

    # Mask features randomly
    x = data.x.clone()
    mask = torch.rand(x.size()) < feat_mask_prob
    x[mask] = 0

    return Data(x=x, edge_index=edge_index)
```

**Masked Feature Prediction:**
```python
# Mask some node features, predict them from neighbors
def mask_features(x, mask_ratio=0.15):
    mask = torch.rand(x.size(0)) < mask_ratio
    masked_x = x.clone()
    masked_x[mask] = 0
    return masked_x, mask
```

---

## 7. Interpretability

### 7.1 GAT Attention Weights

```python
# Get attention weights from GAT
out, (edge_index, attention_weights) = model.conv1(
    x, edge_index, return_attention_weights=True
)

# attention_weights: [num_edges, num_heads]
# Higher weight = more important neighbor
```

### 7.2 GNNExplainer

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
)

# Explain prediction for node 0
explanation = explainer(data.x, data.edge_index, index=0)
print(explanation.node_mask)  # Important features
print(explanation.edge_mask)  # Important edges
```

---

## 8. References

### Darshan
- Darshan Docs: https://darshan.readthedocs.io/
- NERSC Guide: https://docs.nersc.gov/tools/performance/darshan/
- Darshan Paper: "Characterizing and Diagnosing Scalability Issues in HPC I/O" (SC'10)

### Benchmarks
- IOR: https://github.com/hpc/ior
- DLIO: https://github.com/argonne-lcf/dlio_benchmark
- IO500: https://io500.org/

### GNN
- GAT Paper: https://arxiv.org/abs/1710.10903
- GATv2: https://arxiv.org/abs/2105.14491
- PyG: https://pytorch-geometric.readthedocs.io/
- GNNExplainer: https://arxiv.org/abs/1903.03894

### Related Work
- Drishti: https://github.com/hpc-io/drishti-io
- AIIO: https://github.com/hpc-io/aiio
- WisIO (ICS'25): Multi-bottleneck rule-based detection
