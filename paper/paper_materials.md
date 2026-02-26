# SC 2026 Paper Materials

> **Living document**: Updated after each project milestone.
> **Last updated**: 2026-02-26 | Phase: Preprocessing pipeline complete
> **Convention**: Each section maps to a paper section. Facts are tagged with source.

---

## 1. Dataset Description

### 1.1 Source and Scale
- **Source**: ALCF Polaris supercomputer (HPE Apollo Gen10+, 560 nodes, A100 GPUs)
- **Collection**: Darshan instrumentation logs, anonymized and published
- **DOI**: 10.5281/zenodo.15052603
- **Total logs**: 1,397,293 `.darshan` files
- **Successfully parsed**: 1,397,218 (99.995% success rate)
- **Failed**: 75 files (0.005%) — all large/partial logs where PyDarshan C library fails
- **Date range**: 2024-04-24 to 2026-02-24 (~22 months)
- **Unique job IDs**: 122,626
- **Granularity**: Per-application-execution (one row per `.darshan` file, `--total` aggregation across file records within each log)

### 1.2 Feature Space
- **Total columns**: 156 (147 features + 9 info/metadata)
- **Raw counters**: 136 (69 POSIX int, 17 POSIX float, 24 MPI-IO int, 9 MPI-IO float, 10 STDIO int, 7 STDIO float)
- **Indicators**: 8 (has_posix, has_mpiio, has_stdio, has_hdf5, has_pnetcdf, has_apmpi, has_heatmap, is_shared_file)
- **Metadata features**: 3 (nprocs, runtime_seconds, num_files)
- **Output size**: 140 MB (Parquet, compressed)

### 1.3 Module Distribution
| Module Combination | Count | Percentage |
|--------------------|-------|------------|
| STDIO,APMPI,HEATMAP | 546,786 | 39.1% |
| POSIX,STDIO,APMPI,HEATMAP | 443,750 | 31.8% |
| (no I/O modules) | 242,783 | 17.4% |
| APMPI only | 108,338 | 7.8% |
| POSIX,MPI-IO,STDIO,APMPI,HEATMAP | 38,710 | 2.8% |

- **Has POSIX**: 35.6% — only ~1/3 of logs use POSIX file I/O
- **Has STDIO**: 74.3% — dominant interface (Python/ML frameworks use stdio)
- **Has MPI-IO**: 2.9% — only traditional parallel HPC codes
- **No I/O modules**: 17.4% — jobs that did no instrumented I/O

### 1.4 Workload Diversity
| Metric | Value |
|--------|-------|
| nprocs median | 4 |
| nprocs mean | 11.5 |
| nprocs max | 24,800 |
| nprocs P25-P75 | 1-8 |
| nprocs P99 | 128 |
| Single-process jobs | 37.8% |
| Runtime median | 0.13 min (8 sec) |
| Runtime mean | 11.1 min |
| Runtime max | 82.4 hours |
| Total I/O volume | 46,172 TB |
| Zero-I/O logs | 25.6% |

**Key insight**: Polaris is GPU-heavy (A100s), so most jobs are small-nprocs (1-8) AI/ML workloads, not large-nprocs traditional HPC. This is reflected in STDIO dominance over MPI-IO.

### 1.5 Workload Type Classification

#### 1.5.1 Heuristic Classification Results
| Type | Count | Percentage | Signal |
|------|-------|------------|--------|
| Minimal I/O | 647,824 | 46.4% | Total bytes < 1 MB |
| Likely AI/ML | 552,642 | 39.6% | STDIO + no MPI-IO + small reads |
| Metadata-heavy | 159,412 | 11.4% | >1000 opens, <100 MB total |
| Traditional HPC | 36,853 | 2.6% | MPI-IO present, >1 MB total |
| Other | 487 | 0.0% | Unclassified |

#### 1.5.2 I/O Signatures Distinguishing AI/ML from Traditional HPC
Based on Paul et al. (MASCOTS 2021), Lewis et al. (ACM Computing Surveys 2025), and tf-Darshan (Chien et al., CLUSTER 2020):

| Feature | AI/ML (Training) | Traditional HPC (Simulation) |
|---------|-------------------|------------------------------|
| `read_ratio` | > 0.7 (read-intensive data loading) | < 0.3 (checkpoint-heavy writes) |
| `small_read_ratio` | High (many <1KB to 100KB reads) | Low (large contiguous blocks >1MB) |
| `has_stdio` | 1 (Python frameworks use stdio) | 0 (rare for Fortran/C MPI codes) |
| `has_mpiio` | 0 (PyTorch/TF use POSIX directly) | 1 (collective parallel I/O) |
| `opens_per_mb` | High (many file opens per MB) | Low (few large files) |
| `seq_read_ratio` | Low (shuffled data loading) | High (sequential I/O) |
| `num_files` | High (1000s of data files) | Low (1-10 shared files) |
| `is_shared_file` | 0 (per-worker files) | Often 1 (shared checkpoint files) |

#### 1.5.3 Classification Methodology for Paper
**Approach**: Heuristic classification validated by unsupervised clustering (following Bang et al., SNTA 2020).
- Heuristic rules provide interpretability and comparability with prior work
- K-means/GMM clustering on standardized features provides independent statistical validation
- Cross-tabulation shows agreement between heuristic and cluster assignments
- Use language "workloads exhibiting I/O patterns consistent with data-loading ML frameworks" (not definitive claims)
- Acknowledge limitations: anonymized exe names prevent keyword-based confirmation; Python scripts may share similar I/O patterns

#### 1.5.4 Why This Matters for the Paper
1. **Dataset novelty**: Prior work (AIIO on Cori, Paul et al. on Summit) studied CPU-era systems. Polaris is among the first GPU-era I/O datasets with AI-dominated workload mix.
2. **Methodological innovation**: Unlike Paul et al. who used 42 ML keywords on executable names (not possible on anonymized Polaris data), our classification uses only I/O behavioral signatures — more generalizable to any production environment.
3. **Diverse evaluation**: Enables reporting bottleneck detection results separately per workload type (AI/ML vs traditional HPC), addressing a gap in AIIO which was evaluated only on traditional patterns.
4. **Training strategy**: Use workload type as a soft feature in ML models (not separate models). Stratify train/test splits by workload type to ensure representative evaluation.

#### 1.5.5 Polaris Dataset Uniqueness vs Prior Work
| Dataset | System | GPUs? | Years | AI Workloads | Exe Names | Scale |
|---------|--------|-------|-------|--------------|-----------|-------|
| Ours (Polaris) | ALCF, A100 GPUs | Yes | 2024-2026 | ~40% | Anonymized | 1.37M logs |
| Paul et al. (Summit) | OLCF, V100 GPUs | Yes | 2018-2021 | ~3% | Available | ~800K jobs |
| AIIO (Cori) | NERSC, KNL/Haswell | No | 2017-2020 | Minimal | Available | ~1M+ jobs |
| Bang et al. (Cori) | NERSC, KNL/Haswell | No | 2017-2019 | Minimal | Available | Subset |

**Key positioning**: "To our knowledge, this is the first I/O bottleneck detection study on production Darshan logs from a GPU-accelerated leadership computing facility, capturing the emerging AI-dominated workload mix of modern HPC."

### 1.6 Feature Sparsity (Pre-Cleaning)
- **80 features** are >95% zero (very sparse — mostly MPI-IO and STDIO counters that are 0 for non-MPI/non-STDIO jobs)
- **50 features** are 50-95% zero (moderately sparse)
- **Only 2 features** are <10% zero (truly dense)
- This high sparsity motivates careful preprocessing: zero-inflation handling, potential feature selection

### 1.7 Data Cleaning and Filtering

#### 1.7.1 Filtering Strategy (Evidence-Based)
Filtering thresholds determined by threshold sensitivity analysis on the full 1.4M dataset,
informed by literature review of AIIO (HPDC'23), Snyder et al. (CUG'25), Paul et al. (MASCOTS'21),
Bang et al. (SNTA'20), Luu et al. (HPDC'15), Bez et al. (CSUR'23), and Lewis (NIU'21).

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Require POSIX module | true | POSIX provides 86+ features (access patterns, size histograms, alignment) needed for bottleneck classification. STDIO has only 17 counters with no access-pattern resolution (Bez et al. CSUR'23: STDIO addressed by only 3.42% of I/O analysis literature). |
| Min duration | 1 second | Filters sub-second system probes/login shells. Retains legitimate burst I/O (33K jobs in 1-10s range with median 58.6 MB at 13.4 MB/s). |
| Min POSIX bytes | 1024 bytes | Sharp cliff at 0-512 bytes: 332K POSIX jobs have exactly zero POSIX bytes (metadata-only: open configs/libraries via POSIX, real I/O via STDIO). Stable from 512 to 65K (only 11K difference). |
| Min POSIX ops | 1 operation | At least one actual read or write (not just file opens). |

#### 1.7.2 Cleaning Breakdown
| Step | Remaining | Removed | Cumulative % Removed |
|------|-----------|---------|---------------------|
| Raw dataset | 1,397,216 | — | — |
| Require POSIX | 497,327 | 899,889 (64.4%) | 64.4% |
| Min duration >= 1s | 429,835 | 67,492 (4.8%) | 69.2% |
| Min bytes >= 1024 | 131,151 | 298,684 (21.4%) | 90.6% |
| Min ops >= 1 | 131,151 | 0 (0%) | 90.6% |
| **Final dataset** | **131,151** | **1,266,065** | **90.6%** |

#### 1.7.3 Why 90.6% Removal is Expected (Not Aggressive)
- **64.4% lack POSIX module**: Polaris is GPU-dominated; majority workloads use STDIO (Python frameworks). Snyder et al. (CUG'25) report STDIO data volume exceeds POSIX+MPI-IO combined on Polaris. This is the AI-era HPC reality.
- **332K POSIX jobs have zero POSIX bytes**: These jobs open 3,664 files (median) via POSIX for config/library loading but transfer all real data through STDIO. Correctly excluded — POSIX file opens without data transfer are not I/O bottleneck candidates.
- **Literature comparison**:
  - Snyder CUG'25: 33.6% of Polaris logs have zero I/O across ALL modules
  - Luu et al. (HPDC'15): On Edison, 75% of jobs transfer < 3 GB total
  - Paul et al. (MASCOTS'21): Only 8.4% of Summit Darshan logs matched ML-keyword filter
  - AIIO (HPDC'23): Applied no filtering — but on Cori where Lustre counters provide richer features

#### 1.7.4 STDIO-Only AI Workloads — Framing for Paper
> "We analyze all Darshan-logged jobs with sufficient I/O instrumentation resolution for bottleneck classification. On Polaris, 35.6% of jobs access the POSIX interface, providing the access-pattern, size-distribution, and alignment counters required for fine-grained I/O diagnosis. The remaining 64.4% — predominantly AI/ML training jobs using STDIO — lack these diagnostic counters. Our feature extraction includes all three modules (POSIX, MPI-IO, STDIO), enabling model inference on any job type, though diagnostic precision is reduced for STDIO-only workloads."

Key citations:
- Bez et al. (CSUR'23): STDIO is addressed by only 3.42% of surveyed I/O analysis papers — a literature gap
- Snyder et al. (CUG'25): STDIO total data volume exceeds POSIX+MPI-IO on Polaris
- Paul et al. (MASCOTS'21): Analyzed POSIX+MPI-IO+STDIO jointly for ML workloads
- Bang et al. (SNTA'20): Kept STDIO-dominated cluster as valid workload class

### 1.8 Preprocessed Dataset Statistics

#### 1.8.1 After Cleaning (131,151 rows)
| Metric | Value |
|--------|-------|
| Total rows | 131,151 |
| Total columns | 195 (156 raw + 39 derived) |
| Feature columns | 186 |
| Metadata columns | 9 |
| Temporal range | 2024-04-24 to 2026-02-24 |

#### 1.8.2 Module Presence (After Cleaning)
| Module | Count | Percentage |
|--------|-------|------------|
| has_posix | 131,151 | 100.0% |
| has_stdio | 130,636 | 99.6% |
| has_apmpi | 129,864 | 99.0% |
| has_heatmap | 131,151 | 100.0% |
| has_mpiio | 35,978 | 27.4% |

#### 1.8.3 Runtime Distribution (After Cleaning)
| Range | Count | Percentage |
|-------|-------|------------|
| 1-10 seconds | 33,553 | 25.6% |
| 10-60 seconds | 33,408 | 25.5% |
| > 60 seconds | 64,190 | 48.9% |
| Median runtime | 53 seconds | — |

#### 1.8.4 Train/Val/Test Splits (Temporal)
| Split | Rows | Percentage |
|-------|------|------------|
| Train | 91,807 | 70.0% |
| Validation | 19,672 | 15.0% |
| Test | 19,672 | 15.0% |

#### 1.8.5 EDA Highlights (186 Features)
- **125 redundant pairs** (Spearman |rho| > 0.90) — expected for Darshan counters
- **21 sparse features** (>99% zeros) — MMAPS, FDSYNCS, VARIANCE, SPLIT, NB_READS
- **Feature groups**: 12 volume, 39 count, 20 histogram, 24 top4, 25 timing, 8 timestamp, 4 categorical, 2 rank_id, 2 conditional_size, 8 indicator, 23 ratio, 14 ratio_unbounded, 3 derived_absolute, 2 metadata

### 1.9 Normalization Strategy

#### 1.9.1 Approach: Group-Specific Normalization
Rather than applying a single transform to all features, we group the 186 features by
statistical properties and apply the most appropriate normalization per group. This is
necessary because Darshan features span radically different domains: byte counters range
0 to 10^14, bounded ratios live in [0,1], and binary indicators are {0,1}.

**Pipeline order** (prevents data leakage):
1. Clean (Stage 2) → Engineer derived features (Stage 3) → EDA (Stage 4)
2. Temporal split into train/val/test (70/15/15)
3. Fit scalers on **training set only**
4. Transform val and test with pre-fitted scalers

#### 1.9.2 Normalization Methods per Feature Group

| Group | # Features | Method | Rationale |
|-------|-----------|--------|-----------|
| Volume (bytes) | 12 | log1p + RobustScaler | Heavy-tailed (skew~109, 0 to 10^14); RobustScaler centers by median, scales by IQR — resistant to extreme outliers |
| Count (operations) | 39 | log1p + RobustScaler | Heavy-tailed (skew~98, 0 to 10^9); 79% median zero fraction |
| Timing (seconds) | 25 | log1p + RobustScaler | Heavy-tailed (skew~67); cumulative and max single-op times |
| Histogram (size bins) | 20 | log1p | Sparse integer counts, heavy-tailed; log1p compresses without centering |
| Top-4 (access/stride) | 24 | log1p | Mixed sizes and counts, highly sparse (73% median zeros) |
| Conditional Size | 2 | log1p | Paired with max timing, moderate tails |
| Ratio Unbounded | 14 | log1p | Bandwidth, avg size, opens/MB — unbounded positive values |
| Derived Absolute | 3 | log1p | io_duration, dominant_access_size, num_files |
| Metadata | 2 | log1p | nprocs, runtime_seconds — moderate tails |
| Ratio (bounded) | 23 | none | Already [0,1]; normalization would distort interpretation |
| Indicator | 8 | none | Binary {0,1}; no transformation needed |
| Timestamp | 8 | none | Absolute Unix timestamps; used only for derived features |
| Categorical | 4 | none | MODE, ALIGNMENT constants; not continuous |
| Rank ID | 2 | none | Integer rank indices; not performance metrics |

**120 features transformed** (log1p or log1p + RobustScaler), **66 features unchanged**.

#### 1.9.3 RobustScaler Details
- **Formula**: z = (log1p(x) - median) / IQR, where IQR = Q75 - Q25
- **Why RobustScaler over StandardScaler**: Heavy-tailed distributions with extreme outliers. StandardScaler uses mean/std which are distorted by outliers. RobustScaler uses median/IQR which are stable.
- **3 fitted scalers**: volume (12 features), count (39 features), timing (25 features). Saved in `data/processed/scalers.pkl`.
- **Known behavior**: For zero-inflated features (e.g., POSIX_F_MAX_WRITE_TIME with 35% zeros), IQR can be small (~0.009), producing large scaled values for outliers. Tree models handle this natively; for neural models, QuantileTransformer is recommended as alternative.

#### 1.9.4 Dual-Track Strategy (for Model Training)
- **Tree-based models** (XGBoost, LightGBM, RF): Use log1p only (trees are invariant to monotone transforms; RobustScaler adds no benefit)
- **Neural models** (MLP, Transformer): Use full log1p + RobustScaler pipeline, or QuantileTransformer for Gaussian-like output (Gorishniy et al., NeurIPS'21: "On Embeddings for Numerical Features in Tabular Deep Learning")

#### 1.9.5 Feature Exclusion (Post-EDA)
After EDA identified quality issues, 29 features are dropped before normalization:
- **18 constant features** (zero variance on train set): POSIX_MMAPS, POSIX_FDSYNCS, POSIX_MEM_ALIGNMENT, all VARIANCE counters, SPLIT/NB counters, has_posix (always 1 after cleaning), has_hdf5/has_pnetcdf (never present), has_heatmap (always present), rank_bytes_cv/rank_time_cv
- **8 timestamp features**: Absolute within-job timestamps — redundant with io_duration and io_active_fraction derived features. Extreme scale (0 to 189K seconds) and skewness (up to 122) that normalization cannot fix.
- **3 categorical features**: POSIX_RENAMED_FROM (filename hash, std=3.9e17), POSIX_MODE (permission bits), POSIX_FILE_ALIGNMENT (system constant: 0/4096/1048576)

**Result**: 186 → 157 ML-ready features

#### 1.9.6 Key Statistics After Normalization
For the 157 remaining features (train set):
- Median std: 1.07 (well-scaled for ML)
- Features with std in [0.1, 10]: 139/157 (89%)
- Features with p95 in [-10, 10]: 130/157 (83%)
- Zero NaN, zero Inf
- Zero constant features
- Overall range: [-2.85, 1023] (max from POSIX_FASTEST_RANK, an integer rank ID)
- Skewness: mean |skewness| reduced from 77.8 (pre-norm) to 7.6 (post-norm), 10x improvement

### 1.10 Figures and Tables

All preprocessing figures generated by `scripts/generate_paper_figures.py`:
- **Figure: Data Characterization** (`fig_data_characterization.pdf`) — 3 panels: I/O volume distribution, feature sparsity curve, before/after normalization box plots
- **Figure: Temporal Split** (`fig_temporal_split.pdf`) — Timeline with train/val/test regions and weekly job density
- **Figure: Cleaning Funnel** (`fig_cleaning_funnel.pdf`) — Horizontal bar chart of filtering stages and row counts
- **Figure: Feature Correlation** (`fig_feature_correlation.pdf`) — Spearman heatmap of 20 representative features (lower triangle)
- **Figure: Normalization Effect** (`fig_normalization_effect.pdf`) — 2x3 grid showing before/after distributions for 6 features across different groups
- **Table: Dataset Summary** (`tab_dataset_summary.tex`) — Key dataset properties in LaTeX
- **Table: Normalization Methods** (`tab_normalization.tex`) — Per-group normalization in LaTeX

---

## 2. Design Decisions (with Rationale)

### 2.1 Per-Log-File Granularity (not Per-Job)
- **Decision**: Each `.darshan` file = one sample (not aggregated by SLURM job ID)
- **Rationale**: A single SLURM job can run multiple applications with different I/O patterns. Merging them would introduce noise. 76% of jobs have exactly 1 log file anyway.
- **Evidence**: 122,626 unique jobs → 1,397,218 files. Jobs with 100+ files exist (workflow/array jobs). Max: 100,000 files for one job ID.
- **Literature**: AIIO (HPDC'23) uses "job-level" but on NERSC Cori where 1 job ≈ 1 log file. Drishti operates per-log-file.

### 2.2 Raw Feature Extraction (Stage 1)
- **Decision**: Extract ALL 136 raw counters without any transforms or exclusions
- **Rationale**: Defer all normalization, feature selection, and derived feature computation to preprocessing, informed by EDA on the full dataset. Prevents premature decisions.
- **Precedent**: Standard in ML pipelines — separate extraction from preprocessing.

### 2.3 PyDarshan Parser (not CLI)
- **Decision**: Use PyDarshan (Python bindings to libdarshan-util) as primary parser
- **Rationale**: ~10x faster than spawning `darshan-parser` subprocess per file. CLI fallback available for edge cases.
- **Limitation**: 75 files (0.005%) fail with PyDarshan due to large/partial logs. Can be recovered with CLI parser if needed.

### 2.4 Handling Missing Modules
- **Decision**: Zero-fill counters for absent modules (e.g., MPI-IO counters = 0 for non-MPI jobs)
- **Rationale**: Preserves fixed-width feature vector. Module presence captured by indicator features (has_mpiio, etc.). ML models can learn that zero + has_mpiio=0 means "not applicable" vs zero + has_mpiio=1 means "no I/O through that interface."

---

## 3. Processing Pipeline

### 3.1 Batch Extraction Architecture
- **Strategy**: SLURM array jobs (20 nodes) × multiprocessing.Pool (100 workers/node)
- **Total throughput**: ~1.4M files in ~18 minutes wall-clock
- **Per-node rate**: 450-711 files/sec (varies by file complexity)
- **Robustness**: Per-file 120s timeout (signal.alarm), worker recycling (maxtasksperchild=500), atomic writes, checkpoint/resume
- **System**: Delta (NCSA), CPU partition, 128 cores/node, 240GB RAM

### 3.2 Error Analysis
- **Total failures**: 75 / 1,397,293 = 0.005%
- **Root cause**: All `parse_returned_none` — PyDarshan C library cannot handle very large (9-132 MB) or partial-flag Darshan logs
- **Impact**: Negligible. These are extreme outlier jobs (thousands of processes, massive I/O). Not representative of typical workloads.

---

## 4. Experimental Results

### 4.1 Dataset Characterization
<!-- FIGURES: Generated by scripts/generate_paper_figures.py -->
<!-- Run: python scripts/generate_paper_figures.py -->
<!-- Output: figures/preprocessing/*.pdf and *.png -->
See Section 1.10 for full figure inventory. Key figures for dataset section:
- fig_data_characterization: multi-panel overview (volume, sparsity, normalization)
- fig_cleaning_funnel: data cleaning pipeline visualization
- fig_temporal_split: train/val/test temporal split
- tab_dataset_summary.tex: dataset properties table
- tab_normalization.tex: normalization strategy table

### 4.2 Baseline Comparisons
[TODO: After implementing baselines]

### 4.3 Model Performance
[TODO: After training]

### 4.4 LLM Recommendation Quality
[TODO: After RAG implementation]

---

## 5. Related Work Notes

### 5.1 Positioning
| System | Venue | ML? | LLM? | Code Recs? | Our Advantage |
|--------|-------|-----|------|------------|---------------|
| AIIO | HPDC'23 | XGBoost+SHAP | No | No | We add LLM recommendations |
| Drishti | — | No (rules) | No | No | We learn patterns, not hardcode |
| IOAgent | IPDPS'25 | No | Yes (GPT-4) | Yes | We add ML detection (faster, no hallucination) |
| ION | HotStorage'24 | No | Yes | Partial | We have ML+LLM hybrid |
| WisIO | ICS'25 | No (800+ rules) | No | No | We learn from data |
| KORAL | IPDPS'26 | KG+LLM | Yes | SSD-focused | We focus on parallel FS I/O |

### 5.2 Workload Characterization Literature
| Paper | Venue | Method | System | Key Finding |
|-------|-------|--------|--------|-------------|
| Paul et al. | MASCOTS'21/PEVA'22 | 42 ML keywords on exe names | Summit | ML jobs: read-intensive, many small reads, 3% of jobs |
| Bang et al. | SNTA'20 (ACM) | Feature selection + k-means clustering | Cori | 3 optimal clusters from Darshan counters, no exe names needed |
| Betke & Kunkel | ISC HPC'19 | I/O footprinting + ML classification | DKRZ Mistral | Temporal I/O patterns enable job classification |
| Chien et al. | CLUSTER'20 | tf-Darshan fine-grained tracing | Local cluster | TF/PyTorch I/O characterized at operation level |
| Lewis et al. | ACM Comp. Surveys'25 | Survey of ML I/O on HPC | Multiple | "ML workloads perform small reads across random files" |
| Snyder et al. | CUG'25 | Darshan anonymization pipeline | Polaris | Automated log collection + anonymization + Zenodo publication |

### 5.3 Key Differentiator
No published work combines: (1) ML-based I/O pattern detection trained on benchmark-labeled data + (2) LLM-generated code-level fix recommendations grounded in a benchmark knowledge base with measured speedups.

---

## 6. Numbers to Cite in Paper
<!-- Quick-reference for writing. Keep exact numbers here. -->

| What | Value | Source |
|------|-------|--------|
| Total Darshan logs | 1,397,293 | lfs find count |
| Successfully parsed | 1,397,218 | batch extraction output |
| Parse failure rate | 0.005% | 75/1,397,293 |
| Unique job IDs | 122,626 | pandas nunique |
| Feature columns | 147 | feature_extraction.py |
| Info columns | 9 | feature_extraction.py |
| Raw counters | 136 | feature_extraction.py |
| Processing time | ~18 min | SLURM logs |
| Processing rate | 450-711 files/s/node | SLURM logs |
| Output file size | 140 MB | ls -lh |
| Polaris nodes | 560 | ALCF docs |
| Polaris storage | Eagle/Grand Lustre, 100 PiB | ALCF docs |
| Date range | 2024-04-24 to 2026-02-24 | _source_path analysis |
| Has POSIX | 35.6% | module analysis |
| Has STDIO | 74.3% | module analysis |
| Has MPI-IO | 2.9% | module analysis |
| Zero-I/O logs | 25.6% | byte sum analysis |
| Total I/O volume | 46,172 TB | byte sum |
| Single-process jobs | 37.8% | nprocs analysis |
| nprocs median | 4 | nprocs analysis |
| Sparse features (>95% zero) | 80 / 147 | sparsity analysis |
| AI/ML workloads (heuristic) | 39.6% | workload classification |

---

## 7. Figures Checklist
<!-- Track which figures we need and their status -->

| Fig# | Description | Status | Script |
|------|-------------|--------|--------|
| 1 | System architecture diagram | TODO | manual |
| 2 | Dataset temporal distribution (logs/month) | DONE | figures/temporal_distribution.png |
| 3 | Module combination distribution | DONE | figures/module_distribution.png |
| 4 | Job size (nprocs) distribution | DONE | figures/nprocs_distribution.png |
| 5 | I/O volume distribution | DONE | figures/io_volume_distribution.png |
| 5b | Runtime distribution | DONE | figures/runtime_distribution.png |
| 6 | Feature correlation heatmap | DONE | figures/correlation_heatmap.png |
| 7 | Feature sparsity analysis | DONE | figures/feature_sparsity.png |
| 8 | Workload type distribution | DONE | figures/workload_types.png |
| 8b | Read vs Write balance | DONE | figures/read_write_balance.png |
| 8c | t-SNE/UMAP workload clusters | TODO | after clustering validation |
| 8d | Polaris vs prior datasets comparison table | TODO | in paper text |
| 9 | Model comparison (F1 scores) | TODO | after training |
| 10 | SHAP feature importance | TODO | after training |
| 11 | LLM recommendation examples | TODO | after RAG |
| 12 | End-to-end case study | TODO | after full pipeline |

---

## Appendix: Update Log

| Date | Phase | What Changed |
|------|-------|-------------|
| 2026-02-26 | Dataset extraction | Initial creation. Sections 1-3 filled from extraction results. |
| 2026-02-26 | Dataset analysis | Added 9 figures, stats.json, workload classification, sparsity analysis. |
| 2026-02-26 | AI workload research | Added I/O signatures table, classification methodology, dataset uniqueness comparison, workload characterization citations. |
