# HPDC 2026 Revised Roadmap: GNN-Based I/O Bottleneck Detection

## HONEST ASSESSMENT (Read First)

### What IS Novel
- **Topological Resource Contention formalization** - First to model I/O as a graph topology problem
- **Self-Supervised Pre-training on 1.37M Production Logs** - Addresses distribution shift between benchmarks and production via masked edge reconstruction
- **Production-scale GNN for I/O analysis** - Works on 1.37M logs without fine-grained traces
- **Attention-based Root Cause Localization** - Pinpoints which rank-file relationships cause bottlenecks (not just what bottleneck occurred)

### What is NOT Novel (Don't Claim These)
- ~~First GNN for I/O~~ → Avoid this claim due to prior exploratory work
- ~~First multi-bottleneck detection~~ → WisIO (ICS 2025) does this
- ~~First ML for I/O~~ → AIIO (HPDC 2023) does this
- ~~First LLM for I/O~~ → ION/IOAgent do this

### Key Technical Insight
Standard Darshan logs are **aggregated** (no per-operation timestamps). We cannot claim "temporal edges" without fine-grained traces. Instead, we use **structural edges** based on resource sharing, with **concurrency weights** as edge attributes.

---

## Part 1: Updated Competitive Landscape

| Tool | Type | Multi-Bottleneck? | Graph Topology? | Root Cause Localization? | Source |
|------|------|-------------------|-----------------|-------------------------|--------|
| **AIIO** | ML (XGBoost) | Unclear | **No** | SHAP (post-hoc) | [HPDC'23](https://dl.acm.org/doi/10.1145/3588195.3592986) |
| **Drishti** | Rule-based | **No** | **No** | Rule-based | [ISC'23](https://jeanbez.gitlab.io/isc23/) |
| **WisIO** | Rule-based | **Yes** (800+ rules) | **No** | Rule-based | [ICS'25](https://github.com/grc-iit/wisio) |
| **ION** | LLM | Partial | **No** | Natural language | [HotStorage'24](https://dl.acm.org/doi/10.1145/3655038.3665950) |
| **IOAgent** | LLM + RAG | Partial | **No** | Natural language | [IPDPS'25](https://ieeexplore.ieee.org/document/11078545/) |
| **IOGraphNet** | GNN (ML) | **Yes** (8 classes) | **Yes** | **Attention weights (intrinsic)** | - |

**Key Differentiator:**
- WisIO requires 800+ hand-crafted rules. IOGraphNet learns latent relationships automatically.
- AIIO uses post-hoc explainers (SHAP). IOGraphNet uses intrinsic attention weights for root cause localization.
- LLM approaches are slow (seconds) and expensive. IOGraphNet is fast (<10ms) and deterministic.

---

## Part 2: WHY GRAPH? (Critical for Reviewers)

### The Core Argument: Topological Resource Contention

**Critical insight:** Standard Darshan logs are **aggregated** (counters, not traces). We cannot know exact operation timestamps. But we CAN construct a **resource-sharing topology** that reveals bottleneck patterns.

```
FLAT APPROACH (AIIO, Drishti):
[File A features] → Model → "File A is slow"
[Rank 0 features] → Model → "Rank 0 is slow"
(No connection - WHY is Rank 0 slow?)

TOPOLOGY APPROACH (IOGraphNet):
[Rank 0] ──edge──► [File A] ◄──edge── [Rank 1]
                      │
                   high degree = contention!

IOGraphNet learns: "Rank 0 is slow BECAUSE it connects to File A, which has high degree"
```

### Why Topology Reveals Bottlenecks (Without Fine-Grained Traces)

| Graph Pattern | Bottleneck Revealed |
|---------------|---------------------|
| **N-to-1 (Star)**: N ranks → 1 shared file | Lock contention, serialization |
| **1-to-N**: 1 rank → N files | Metadata stress, file-per-process explosion |
| **Disconnected subgraphs** | Independent I/O (potential for collective) |
| **High-degree node** | Hot file, bandwidth bottleneck |

### Concrete Examples

#### Example 1: Data Imbalance Detection

**Flat approach:**
- Rank 0: 10GB writes → "High I/O volume"
- Rank 42: 100MB writes → "Normal I/O volume"
- **No connection made**

**Topology approach:**
- Same-file edge connects Rank 0 and Rank 42 records
- GNN learns: high variance across connected nodes = imbalance
- **Attention weight on Rank 0 → File edge = 0.92** (root cause!)

#### Example 2: Interface Misuse (Missing Collective)

**Flat approach:**
- 64 POSIX records for same file
- Each looks like normal independent I/O

**Topology approach:**
- Same-file edges form a star: 64 ranks → 1 file
- High in-degree on file node = "64 independent accesses to shared file"
- GNN learns: star pattern + zero collective calls = interface misuse

#### Example 3: Metadata Storm

**Flat approach:**
- 10,000 POSIX records with small byte counts
- Each looks like "small file access"

**Topology approach:**
- 1-to-N pattern: 1 rank → 10,000 files
- GNN learns: high out-degree on rank node = metadata explosion

### Graph Construction (From Aggregated Logs)

```
For each Darshan log (job):

NODES:
├── File nodes (one per file-module combination)
│   ├── Features: ~60 Darshan counters
│   └── Type: POSIX, MPI-IO, STDIO

STRUCTURAL EDGES (determinable from aggregated data):
├── same_rank: Files accessed by same MPI rank
└── same_file: Different ranks/modules accessing same file

EDGE ATTRIBUTES:
└── concurrency_weight = |time_window_i ∩ time_window_j| / |time_window_i ∪ time_window_j|
    (From aggregated start/end times - indicates intensity, not exact overlap)
```

**Important:** We do NOT claim "temporal contention edges" (would require traces). We claim "structural resource sharing" with concurrency intensity as an attribute.

### What to Tell Reviewers

**Reviewer Question:** "Why do you need a graph? Can't you just concatenate features?"

**Your Answer:**
> "Flat-feature approaches like AIIO and Drishti treat each file independently, missing critical relationships. For example:
> 1. **Load imbalance** emerges from relationships BETWEEN files/ranks, not individual files
> 2. **Missing collective I/O** is detected by seeing multiple independent accesses to the SAME file
> 3. **Cross-module patterns** (POSIX vs MPI-IO) require connecting records for the same file
> 4. **Contention** requires knowing which jobs overlap temporally
>
> These relationships are naturally modeled as a graph. Our GNN learns to aggregate information across connected nodes, capturing patterns that flat models miss."

**Empirical Evidence (You Must Show):**
- Ablation: GNN vs MLP with same features → GNN should be better
- Case study: Show a bottleneck that requires relationships to detect

---

## Part 3: Architecture with Root Cause Localization

**Key feature:** Not just classification, but **which rank-file relationships caused the bottleneck**.

### Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class IOGraphNet(nn.Module):
    """GNN for multi-label I/O bottleneck classification with attention-based localization."""

    def __init__(self, in_features, hidden_dim=64, num_classes=8):  # 8 classes now
        super().__init__()

        # Two GAT layers (attention for root cause localization)
        self.conv1 = GATConv(in_features, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        # Multi-label classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return torch.sigmoid(self.classifier(h_graph))

    def forward_with_attention(self, x, edge_index, batch):
        """Return predictions AND attention weights for root cause localization."""
        h, (_, attn1) = self.conv1(x, edge_index, return_attention_weights=True)
        h = F.elu(h)
        h, (_, attn2) = self.conv2(h, edge_index, return_attention_weights=True)
        h = F.elu(h)
        h_graph = global_mean_pool(h, batch)
        pred = torch.sigmoid(self.classifier(h_graph))
        return pred, {'layer1_attn': attn1, 'layer2_attn': attn2}

# Training
model = IOGraphNet(in_features=60, hidden_dim=64, num_classes=8)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### The "Killer Feature": Attention-Based Root Cause Localization

```python
def get_root_cause(model, graph, threshold=0.5):
    """
    After prediction, identify WHICH rank-file edges caused the bottleneck.
    This is the key differentiator from flat-feature models.
    """
    pred, attention = model.forward_with_attention(graph.x, graph.edge_index, graph.batch)

    # Get predicted bottlenecks
    bottlenecks = (pred > threshold).nonzero()

    # Get top-k attention edges (root causes)
    attn_weights = attention['layer2_attn'].squeeze()
    top_k_edges = attn_weights.topk(k=5).indices

    # Map edges back to rank-file pairs
    root_causes = []
    for edge_idx in top_k_edges:
        src, dst = graph.edge_index[:, edge_idx]
        root_causes.append({
            'source_node': src.item(),
            'target_node': dst.item(),
            'attention_weight': attn_weights[edge_idx].item(),
            'interpretation': f"High attention ({attn_weights[edge_idx]:.2f}) indicates this rank-file interaction contributed to bottleneck"
        })

    return {
        'predicted_bottlenecks': bottlenecks,
        'root_causes': root_causes
    }

# Example output:
# {
#   'predicted_bottlenecks': ['Data Imbalance'],
#   'root_causes': [
#     {'source': 'Rank 42', 'target': 'checkpoint.h5', 'attention': 0.92,
#      'interpretation': 'Rank 42 interaction with checkpoint.h5 caused imbalance'}
#   ]
# }
```

### 8 Bottleneck Classes (Consolidated for Robust Detection)

**Rationale:** 13 classes was too fine-grained for ~2,000 training samples. We merged similar classes that are hard to distinguish from aggregated counters.

| ID | Bottleneck | Detection Criteria | Merged From |
|----|------------|-------------------|-------------|
| 0 | **Metadata Storm** | meta_time > data_time OR opens/ops > 10 | Metadata Heavy + Excessive Opens |
| 1 | **Small I/O** | >80% operations <1KB | Small I/O + Misaligned |
| 2 | **Data Imbalance** | CV(rank_bytes) > 0.5 OR max/median > 2 | Load Imbalance + Stragglers |
| 3 | **Write Saturation** | High bytes, low BW vs peak | Write bottleneck |
| 4 | **Read Saturation** | High bytes, low BW vs peak | Read bottleneck |
| 5 | **Interface Misuse** | Shared file, high indep ops, zero collective | Missing Collective |
| 6 | **File-per-Process** | files_created > 100 × ranks | Excessive file creation |
| 7 | **Healthy** | High BW, balanced access | No Bottleneck |

**Dropped classes:** Random Access (hard to detect reliably), Contention/OST Imbalance (requires system-level info not in Darshan), Checkpoint/Data Loading (merged into saturation patterns).

---

## Part 3.5: Self-Supervised Pre-training Strategy (NEW - Critical)

### The Problem: Distribution Shift

Training only on benchmark data is RISKY:

| Data Source | Characteristics | Problem |
|-------------|-----------------|---------|
| **Benchmarks (IOR, DLIO)** | Clean, predictable, controlled | "Toy data" - not representative |
| **Production (Polaris)** | Noisy, complex, diverse | Real patterns but no labels |

**Reviewer will ask:** "How do you know your model works on production logs if you never trained on them?"

### The Solution: Foundation Model Approach

We use the 1.37M unlabeled Polaris logs for **pre-training**, not just generalization testing.

```
Stage 1: Self-Supervised Pre-training (1.37M Unlabeled Logs)
├── Task: Masked Edge Reconstruction
├── Goal: Learn "what HPC I/O graphs look like"
└── No labels needed!

Stage 2: Supervised Fine-tuning (2,000 Labeled Logs)
├── Task: Multi-label Classification
├── Goal: Learn "what bottleneck names mean"
└── Uses pre-trained encoder weights
```

### Technical Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, InnerProductDecoder
from torch_geometric.utils import negative_sampling

class IOGraphNetPretraining(nn.Module):
    """Self-supervised pre-training via masked edge reconstruction."""

    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        # Same encoder as IOGraphNet
        self.conv1 = GATConv(in_features, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        # Decoder for edge reconstruction
        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        """Encode graph to node embeddings."""
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        return h

    def decode(self, z, pos_edge_index, neg_edge_index):
        """Reconstruct edge probabilities."""
        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        return pos_pred, neg_pred

def pretrain_step(model, data, optimizer, mask_ratio=0.15):
    """One pre-training step with masked edge reconstruction."""
    model.train()

    # Randomly mask 15% of edges
    num_edges = data.edge_index.size(1)
    num_mask = int(num_edges * mask_ratio)
    perm = torch.randperm(num_edges)
    mask_indices = perm[:num_mask]
    keep_indices = perm[num_mask:]

    # Split edges
    masked_edges = data.edge_index[:, mask_indices]
    visible_edges = data.edge_index[:, keep_indices]

    # Encode with visible edges only
    z = model.encode(data.x, visible_edges)

    # Sample negative edges
    neg_edges = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_mask
    )

    # Predict masked edges
    pos_pred, neg_pred = model.decode(z, masked_edges, neg_edges)

    # Binary cross-entropy loss
    pos_loss = -torch.log(pos_pred + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
    loss = pos_loss + neg_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Pre-training loop
pretrain_model = IOGraphNetPretraining(in_features=60, hidden_dim=64)
pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.001)

for epoch in range(50):  # Pre-train for 50 epochs
    for batch in polaris_loader:  # 1.37M logs
        loss = pretrain_step(pretrain_model, batch, pretrain_optimizer)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### Fine-tuning

```python
class IOGraphNet(nn.Module):
    """Full model for classification, initialized from pre-trained weights."""

    def __init__(self, pretrained_encoder, hidden_dim=64, num_classes=8):
        super().__init__()
        # Copy pre-trained encoder weights
        self.conv1 = pretrained_encoder.conv1
        self.conv2 = pretrained_encoder.conv2

        # New classification head (random init)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return torch.sigmoid(self.classifier(h_graph))

# Initialize from pre-trained
model = IOGraphNet(pretrained_encoder=pretrain_model, num_classes=8)

# Fine-tune on labeled benchmarks
for epoch in range(100):
    for batch in benchmark_loader:  # 2,000 labeled logs
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.binary_cross_entropy(pred, batch.y)
        loss.backward()
        optimizer.step()
```

### Why This Works

To predict that Rank 0 → File A edge was masked, the model must learn:
1. **Local patterns**: What kind of connections are typical for this node
2. **Global structure**: Star patterns (contention) vs file-per-process patterns
3. **HPC physics**: Collective I/O patterns, rank locality, striping effects

This is the same structural understanding needed for bottleneck detection!

### Paper Framing

**Old approach (weak):**
> "We train on 2,000 benchmark logs and test generalization on 1.37M Polaris logs."

**New approach (strong):**
> "We employ a two-stage training strategy inspired by foundation models. First, we pre-train on 1.37M unlabeled Polaris logs using masked edge reconstruction, learning the structural patterns of real HPC I/O. Then, we fine-tune on 2,000 labeled benchmark logs for classification. This addresses the distribution shift between controlled benchmarks and production workloads."

### Expected Results

| Training Strategy | Micro-F1 | Robustness on Polaris |
|-------------------|----------|----------------------|
| Supervised only | ~0.81 | Lower (distribution shift) |
| Pre-trained + Fine-tuned | **~0.86** | **Higher (seen production patterns)** |

### References to Add

- [Kipf & Welling (2016)](https://arxiv.org/abs/1611.07308) - Variational Graph Auto-Encoders
- [MaskGAE (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599546) - Masked Graph Modeling
- [Strategies for Pre-training GNNs (ICLR 2020)](https://openreview.net/forum?id=HJlWWJSFDH) - Pre-training strategies

---

## Part 4: Ground Truth Strategy

### Use Benchmarks (8 Consolidated Classes)

| Benchmark | Bottleneck Class | Target Logs | Label |
|-----------|------------------|-------------|-------|
| mdtest (stat-heavy) | Metadata Storm | 250 | 0 |
| IOR small (64B-1KB) | Small I/O | 250 | 1 |
| Custom imbalanced MPI | Data Imbalance | 250 | 2 |
| IOR large sequential write | Write Saturation | 200 | 3 |
| IOR large sequential read | Read Saturation | 200 | 4 |
| IOR POSIX on shared file | Interface Misuse | 200 | 5 |
| mdtest file-per-proc | File-per-Process | 200 | 6 |
| IOR optimized collective | Healthy | 300 | 7 |
| **Total** | | **~1,850 logs** | |

### Train/Test Split

```
Benchmark logs (~1,850):
├── Train: 70% (~1,300 logs)
├── Validation: 15% (~275 logs)
└── Test: 15% (~275 logs)

Polaris logs (1.37M):
└── Generalization demo only (no labels needed for training)
    - Use t-SNE to show learned embeddings cluster by bottleneck type
    - Use agreement with Drishti as weak validation
```

---

## Part 5: Evaluation Strategy

### Primary Metrics (Benchmark Test Set)

| Metric | Formula | Target |
|--------|---------|--------|
| Hamming Loss | Wrong labels / total labels | <0.15 |
| Micro-F1 | Overall precision/recall | >0.80 |
| Macro-F1 | Average F1 per class | >0.75 |

### Ablation Studies (CRITICAL)

| Experiment | Purpose | Expected Result |
|------------|---------|-----------------|
| GNN vs MLP | Prove graph helps | GNN > MLP by 5%+ |
| GNN vs XGBoost | Compare with AIIO | GNN > XGBoost |
| With edges vs no edges | Prove edges matter | With > Without |
| Multi-module vs POSIX-only | Prove multi-module helps | Multi > POSIX |

### Baselines to Implement

1. **Structure-Aware XGBoost** - AIIO-style with graph-derived features (see below)
2. **Random Forest** - Same structure-aware features
3. **MLP** - Same structure-aware features
4. **Rule-based** - Implement Drishti-like threshold rules

### Structure-Aware Baseline Features (Critical for Fair Comparison)

**Problem:** Reviewers will ask: "Why compare GNN to XGBoost? That's apples-to-oranges!"

**Solution:** Give XGBoost "structure-aware" features by flattening the graph into statistical summaries:

| Category | Feature | Topological Intuition |
|----------|---------|----------------------|
| **Global** | Total Ops (Read/Write/Meta) | Overall job intensity (AIIO-style) |
| **Global** | Total Bytes (R/W) | Data volume |
| **Graph (Node)** | Max/Mean Rank Degree | Fan-out: metadata stress |
| **Graph (Node)** | Max/Mean File Degree | Fan-in: contention/locking |
| **Graph (Node)** | Degree Std. Dev. | Load imbalance across nodes |
| **Graph (Edge)** | Max Concurrency Weight | Peak contention on single file |
| **Graph (Edge)** | Mean Concurrency Weight | General system pressure |
| **Graph (Edge)** | Edge Weight Skewness | Stragglers vs. uniform I/O |
| **Imbalance** | Gini Coefficient (bytes) | Data distribution inequality |
| **Imbalance** | Max/Mean Time Ratio | Straggler detection |

**Key Insight:** If IOGraphNet outperforms structure-aware XGBoost, it proves **message-passing** (not just structural statistics) is valuable.

```python
def flatten_graph_for_baseline(graph):
    """Convert graph to flat features for XGBoost/MLP baselines."""
    features = {}

    # Global counters (standard)
    features['total_bytes'] = graph.x[:, 0].sum()
    features['total_ops'] = graph.x[:, 1].sum()

    # Degree statistics
    degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    features['degree_min'] = degrees.min()
    features['degree_max'] = degrees.max()
    features['degree_mean'] = degrees.float().mean()
    features['degree_std'] = degrees.float().std()

    # Edge weight statistics (if available)
    if graph.edge_attr is not None:
        features['weight_min'] = graph.edge_attr.min()
        features['weight_max'] = graph.edge_attr.max()
        features['weight_mean'] = graph.edge_attr.mean()
        features['weight_std'] = graph.edge_attr.std()

    # Imbalance metrics
    bytes_per_node = graph.x[:, 0]  # Assuming first feature is bytes
    features['gini_bytes'] = gini_coefficient(bytes_per_node)
    features['max_mean_ratio'] = bytes_per_node.max() / (bytes_per_node.mean() + 1e-8)

    return features
```

### Class Imbalance Handling (Critical for Production Logs)

**Problem:** Production logs have severe class imbalance:
- ~60% "Healthy" or "Small I/O" (common patterns)
- <5% "Write Saturation" (rare bottleneck)

If we train on balanced benchmarks and test on imbalanced production, model may over-predict rare classes.

**Solution:** Use class-balanced weighted BCE loss:

**Loss Function (matches paper):**
```
L = -Σₖ wₖ [ yₖ log(ŷₖ) + (1-yₖ) log(1-ŷₖ) ]
```

where `wₖ = N / (K · nₖ)` is the inverse frequency weight.

```python
def compute_class_weights(labels):
    """Compute inverse frequency weights for class imbalance."""
    # labels: [N, K] binary matrix
    N, K = labels.shape
    pos_counts = labels.sum(dim=0)  # Count of positive for each class
    weights = N / (K * pos_counts + 1e-8)  # Inverse frequency: w_k = N / (K * n_k)
    return weights

# In training loop
pos_weight = compute_class_weights(train_labels)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This ensures rare bottlenecks (Write Saturation) contribute proportionally to the loss.

---

## Part 6: Interpretation (Simple Approach)

### Use GAT Attention Weights

```python
# Get attention weights during forward pass
def forward_with_attention(self, x, edge_index, batch):
    h, (edge_index_1, attn_1) = self.conv1(x, edge_index, return_attention_weights=True)
    h = F.elu(h)
    h, (edge_index_2, attn_2) = self.conv2(h, edge_index, return_attention_weights=True)
    h = F.elu(h)

    h_graph = global_mean_pool(h, batch)
    pred = torch.sigmoid(self.classifier(h_graph))

    return pred, attn_1, attn_2  # Attention weights for interpretation
```

### Interpretation Pipeline

```
1. Predict bottleneck → [small_io=0.92, load_imbalance=0.78]
2. Get attention weights → Edge (File A → File B) has weight 0.85
3. Look at high-attention nodes → File A has 95% small reads
4. Generate explanation → "Small I/O detected: File A (95% reads <1KB)"
```

### Recommendation Mapping (Rule-Based)

| Bottleneck | Recommendation |
|------------|----------------|
| Small I/O | Increase buffer size to 1MB+ |
| Random Access | Sort I/O, use collective operations |
| Missing Collective | Use MPI_File_write_all instead of MPI_File_write |
| Load Imbalance | Redistribute work, use collective I/O |
| Checkpoint Bottleneck | Use async I/O, compress checkpoints |

---

## Part 7: Feasible Implementation Plan

### Phase 1: Data Preparation (Week 1-2)

- [ ] Parse 1000 sample Darshan logs with PyDarshan
- [ ] Extract features (POSIX, MPI-IO, STDIO counters)
- [ ] Implement graph construction
- [ ] Test on sample data

### Phase 2: Benchmark Collection (Week 2-4)

- [ ] Set up IOR, mdtest, DLIO on test cluster
- [ ] Run benchmarks with each bottleneck pattern
- [ ] Collect ~2,000 labeled logs
- [ ] Create train/val/test splits

### Phase 3: Model Development (Week 4-6)

- [ ] Implement IOGraphNet in PyTorch Geometric
- [ ] Train on benchmark data
- [ ] Tune hyperparameters
- [ ] Achieve >80% Micro-F1

### Phase 4: Evaluation (Week 6-8)

- [ ] Implement baselines (XGBoost, RF, MLP)
- [ ] Run ablation studies
- [ ] Generate comparison tables
- [ ] Test on Polaris logs (generalization)

### Phase 5: Paper Writing (Week 8-10)

- [ ] Write paper (11 pages)
- [ ] Create figures (architecture, results, ablation)
- [ ] Build Streamlit demo
- [ ] Internal review

---

## Part 8: Paper Outline

```
1. Introduction
   - I/O bottleneck is critical HPC problem
   - Existing tools miss relationships between files
   - We propose first GNN-based approach

2. Background & Related Work
   - Darshan logging
   - AIIO (flat ML)
   - Drishti/WisIO (rule-based)
   - ION/IOAgent (LLM-based)
   - GNN in HPC (scheduling, not I/O)

3. Approach
   - Graph construction (files as nodes, relationships as edges)
   - GNN architecture (GAT)
   - Multi-label classification

4. Evaluation
   - Dataset: ALCF Polaris (1.37M logs)
   - Ground truth: Benchmarks (2,000 logs)
   - Metrics: Hamming loss, F1
   - Ablation: GNN vs MLP, with/without edges
   - Baselines: XGBoost, RF, Drishti

5. Results
   - GNN outperforms flat models
   - Graph structure improves detection by X%
   - Case study: Bottleneck requiring relationships

6. Discussion & Limitations
   - Limitations: Benchmark-based ground truth
   - Future: LLM for explanation

7. Conclusion
```

---

## Part 9: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GNN doesn't beat baselines | Focus on ablation showing edges help, even if overall accuracy similar |
| Reviewers question "why graph?" | Strong examples in Section 2, ablation study |
| Not enough benchmark data | Start early, can use fewer classes if needed |
| WisIO is too similar | Emphasize: WisIO is rule-based, you use learned GNN |
| Time runs out | Keep it simple - no LLM, no regression |

---

## Part 10: Key Differences from Previous Roadmap

| Aspect | Previous | Revised |
|--------|----------|---------|
| Task | Multi-task (classification + regression + severity) | **Classification only** |
| Novelty claim | First multi-bottleneck | **First GNN** (honest) |
| Competitors | Missing WisIO | **WisIO included** |
| "Why graph?" | Weak justification | **Strong with examples** |
| Complexity | Over-engineered | **Simple and feasible** |

---

## Summary: Your Novel Contribution

**"We present IOGraphNet, a production-scale graph neural network approach for HPC I/O bottleneck detection that formalizes the problem as Topological Resource Contention. To address the distribution shift between controlled benchmarks and production workloads, we employ self-supervised pre-training on 1.37M unlabeled Polaris logs via masked edge reconstruction—learning the structural patterns of real HPC I/O before fine-tuning for classification. Unlike flat-feature methods (AIIO) that treat counters independently, IOGraphNet models the resource-sharing topology between ranks and files—revealing that 'Rank 0 is slow because it connects to high-degree File A.' Unlike WisIO's 800+ hand-crafted rules, IOGraphNet learns latent patterns automatically. Unlike LLM approaches (IOAgent) that are slow and expensive, IOGraphNet provides deterministic, sub-10ms inference. Most critically, IOGraphNet provides attention-based root cause localization: not just 'what bottleneck occurred' but 'which rank-file relationships caused it'—a capability no existing tool provides."**

### Key Claims (Defensible)
1. **Topological Resource Contention** - Novel formalization for aggregated logs
2. **Self-Supervised Pre-training** - Leverages 1.37M unlabeled logs to learn production patterns
3. **Attention-Based Root Cause Localization** - Intrinsic, not post-hoc (unlike SHAP)
4. **Production-Scale** - Works on 1.37M logs without fine-grained traces
5. **Fast & Deterministic** - <10ms inference, no LLM API costs
6. **Fair Baseline Comparison** - Structure-aware XGBoost with graph-derived features proves message-passing value
7. **Class Imbalance Robust** - Weighted loss handles production-like skewed distributions

---

## Part 11: Artifact & Appendix Preparation

### HPDC 2026 Artifact Requirements

| Item | Requirement | Status |
|------|-------------|--------|
| AD/AE Appendix | Optional, up to 2 pages | ⏳ Pending |
| Submit with paper | By Feb 5, 2026 | ⏳ Pending |
| Double-blind | Must not reveal identity | ⏳ Pending |
| Public availability | Encouraged (Zenodo DOI) | ⏳ Pending |

### Artifacts to Prepare

| Artifact | Description | Priority |
|----------|-------------|----------|
| Source code | IOGraphNet implementation | Required |
| Trained models | Pre-trained + fine-tuned weights | Required |
| Benchmark data | Labeled graphs for testing | Required |
| Sample data | Small subset for quick testing | Required |
| Docker image | Reproducible environment | Optional |
| README/INSTALL | Documentation | Required |

### Trained Model Submission

- **Format**: PyTorch `.pt` files
- **Include**: Both pre-trained (1.37M logs) and fine-tuned (benchmark) weights
- **When**: With paper submission (Feb 5) or after acceptance
- **Where**: GitHub + Zenodo (for DOI)

### AD/AE Appendix Content

```
Artifact Description:
- What: Code, models, data
- Where: GitHub URL + Zenodo DOI
- Hardware: GPU requirements
- Software: Python/PyTorch versions

Artifact Evaluation:
- Setup: pip install -r requirements.txt
- Run: python evaluate.py
- Expected: Micro-F1 ~ 0.86
- Time: 5 min (evaluation), 2 hr (training)
```

See PROJECT_PHASES.md Phase 5.5 for detailed checklist.

---

## References

### Must-Cite (Competitors)
1. [AIIO (HPDC'23)](https://dl.acm.org/doi/10.1145/3588195.3592986) - ML baseline
2. [Drishti (ISC'23)](https://jeanbez.gitlab.io/isc23/) - Rule-based
3. [WisIO (ICS'25)](https://github.com/grc-iit/wisio) - Multi-perspective rule-based
4. [IOAgent (IPDPS'25)](https://ieeexplore.ieee.org/document/11078545/) - LLM-based

### Supporting
5. [ION (HotStorage'24)](https://dl.acm.org/doi/10.1145/3655038.3665950) - LLM approach
6. [I/O Access Patterns Survey](https://dl.acm.org/doi/10.1145/3611007) - Background
7. [HPCGCN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9893918/) - GNN in HPC (not I/O)
