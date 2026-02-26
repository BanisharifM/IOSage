# SC 2026 Paper Materials

> **Living document**: Updated after each project milestone.
> **Last updated**: 2026-02-26 | Phase: Dataset extraction complete
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

### 1.5 Workload Type Classification (Heuristic)
| Type | Count | Percentage | Signal |
|------|-------|------------|--------|
| Minimal I/O | 647,824 | 46.4% | Total bytes < 1 MB |
| Likely AI/ML | 552,642 | 39.6% | STDIO + no MPI-IO + small reads |
| Metadata-heavy | 159,412 | 11.4% | >1000 opens, <100 MB total |
| Traditional HPC | 36,853 | 2.6% | MPI-IO present, >1 MB total |
| Other | 487 | 0.0% | Unclassified |

**Can we identify AI workloads?** Yes, heuristically. Polaris is a GPU system, so AI/ML workloads dominate. Signals: (1) STDIO-heavy (Python frameworks), (2) no MPI-IO (PyTorch/TF use POSIX directly), (3) many small reads (data loading), (4) small nprocs (1-8 GPUs). The exe name is hashed (anonymized), so we cannot confirm definitively, but the I/O pattern signature is distinctive.

**Is this useful for the paper?** Yes — it shows our dataset covers diverse workload types, not just one class. The AI/ML vs traditional split is interesting because existing I/O analysis tools (AIIO, Drishti) were designed for traditional HPC. Polaris represents the emerging GPU-centric, AI-heavy workload mix.

### 1.6 Feature Sparsity
- **80 features** are >95% zero (very sparse — mostly MPI-IO and STDIO counters that are 0 for non-MPI/non-STDIO jobs)
- **50 features** are 50-95% zero (moderately sparse)
- **Only 2 features** are <10% zero (truly dense)
- This high sparsity motivates careful preprocessing: zero-inflation handling, potential feature selection

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
<!-- FIGURES: Generated by scripts/analyze_dataset.py -->
[TODO: Reference figure files after running analysis]

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

### 5.2 Key Differentiator
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
