# Darshan Log Dataset Comparison

## AIIO Dataset vs ALCF Polaris Darshan Log Collection

This document compares two HPC I/O characterization datasets derived from Darshan logs.

---

## Overview

| Attribute | AIIO Dataset | ALCF Polaris Collection |
|-----------|--------------|-------------------------|
| **Source System** | NERSC Cori Supercomputer | ALCF Polaris Supercomputer |
| **File System** | Lustre | Lustre (Eagle/Grand), ext4, NFS |
| **Time Period** | ~40 months (historical) | 2024-2026 (ongoing) |
| **Dataset Size** | 6M+ jobs | 1.37M+ logs |
| **Format** | Pre-processed CSV | Raw .darshan files |
| **Normalization** | log10(x + 1) | Raw counter values |
| **Labels/Tags** | Continuous performance score | None (unlabeled) |
| **DOI** | 10.1145/3588195.3592986 | 10.5281/zenodo.15052603 |

---

## System Specifications

### NERSC Cori (AIIO Source)
- **Architecture**: Cray XC40
- **Nodes**: 2,388 Intel Xeon "Haswell" + 9,688 Intel Xeon Phi "Knights Landing"
- **Storage**: Lustre file system (scratch)
- **Peak Performance**: ~30 PFLOPS

### ALCF Polaris (Polaris Collection Source)
- **Architecture**: HPE Apollo 6500 Gen 10+
- **Nodes**: 520 compute nodes
- **CPU**: AMD EPYC Milan 7543P (32 cores, 2.8 GHz)
- **GPU**: 4x NVIDIA A100 per node
- **Storage**:
  - Eagle/Grand: Lustre, 100 PiB, 650 GiB/s (160 OSTs, 40 MDTs)
  - Home: Lustre (non-intensive)
  - Node-local: 3.2 TiB SSD, 6 GiB/s

---

## Data Structure Comparison

### AIIO Dataset Structure
- **Format**: Tabular CSV (one row per job)
- **Features**: 45 I/O counters + 1 target variable
- **Normalization**: All values transformed using `log10(x + 1)`
- **Target Variable**: `tag` - continuous performance score

**Feature Categories in AIIO:**
| Category | Count | Examples |
|----------|-------|----------|
| Job Metadata | 1 | nprocs |
| Lustre Config | 2 | LUSTRE_STRIPE_SIZE, LUSTRE_STRIPE_WIDTH |
| POSIX Operations | 8 | OPENS, READS, WRITES, SEEKS, STATS |
| POSIX Bytes | 2 | BYTES_READ, BYTES_WRITTEN |
| POSIX Sequential | 6 | SEQ_READS, SEQ_WRITES, CONSEC_* |
| POSIX Alignment | 4 | MEM_ALIGNMENT, FILE_ALIGNMENT, *_NOT_ALIGNED |
| POSIX Size Buckets | 9 | SIZE_READ_0_100, SIZE_WRITE_100K_1M, etc. |
| POSIX Stride | 8 | STRIDE1-4_STRIDE, STRIDE1-4_COUNT |
| POSIX Access | 8 | ACCESS1-4_ACCESS, ACCESS1-4_COUNT |

### ALCF Polaris Collection Structure
- **Format**: Raw binary .darshan files
- **Organization**: year/month/day directory hierarchy
- **Modules**: POSIX, STDIO, MPI-IO, APMPI, HEATMAP
- **Anonymization**: UID, exe, file paths hashed

**Available Modules and Counter Counts:**
| Module | Counter Categories | Approximate Counters |
|--------|-------------------|---------------------|
| POSIX | Operations, bytes, timing, access patterns | 70+ |
| MPI-IO | Collective/independent ops, timing, sizes | 65+ |
| STDIO | File operations, bytes, timing | 30+ |
| APMPI | Application MPI metrics | Variable |
| HEATMAP | Temporal I/O distribution | Variable |

---

## Feature Comparison

### Common POSIX Counters (Both Datasets)

| Counter | AIIO | Polaris | Notes |
|---------|:----:|:-------:|-------|
| POSIX_OPENS | Yes | Yes | File open operations |
| POSIX_READS | Yes | Yes | Read operations |
| POSIX_WRITES | Yes | Yes | Write operations |
| POSIX_SEEKS | Yes | Yes | Seek operations |
| POSIX_STATS | Yes | Yes | Stat calls |
| POSIX_BYTES_READ | Yes | Yes | Total bytes read |
| POSIX_BYTES_WRITTEN | Yes | Yes | Total bytes written |
| POSIX_CONSEC_READS | Yes | Yes | Consecutive reads |
| POSIX_CONSEC_WRITES | Yes | Yes | Consecutive writes |
| POSIX_SEQ_READS | Yes | Yes | Sequential reads |
| POSIX_SEQ_WRITES | Yes | Yes | Sequential writes |
| POSIX_RW_SWITCHES | Yes | Yes | Read/write switches |
| POSIX_MEM_ALIGNMENT | Yes | Yes | Memory alignment |
| POSIX_FILE_ALIGNMENT | Yes | Yes | File alignment |
| POSIX_MEM_NOT_ALIGNED | Yes | Yes | Unaligned memory ops |
| POSIX_FILE_NOT_ALIGNED | Yes | Yes | Unaligned file ops |
| POSIX_SIZE_READ_* | Yes | Yes | Read size histograms |
| POSIX_SIZE_WRITE_* | Partial | Yes | Write size histograms |
| POSIX_STRIDE1-4_* | Yes | Yes | Stride patterns |
| POSIX_ACCESS1-4_* | Yes | Yes | Access patterns |

### Counters Only in Polaris Collection

| Counter Category | Examples |
|------------------|----------|
| Timing Metrics | F_READ_TIME, F_WRITE_TIME, F_META_TIME |
| Timestamps | F_OPEN_START_TIMESTAMP, F_CLOSE_END_TIMESTAMP |
| Rank Statistics | FASTEST_RANK, SLOWEST_RANK, VARIANCE_RANK_* |
| Additional Ops | DUPS, MMAPS, FSYNCS, FDSYNCS, RENAME_* |
| MPI-IO Module | COLL_READS, INDEP_WRITES, NB_READS, VIEWS |
| STDIO Module | FLUSHES, FDOPENS |

### Counters Only in AIIO Dataset

| Counter | Notes |
|---------|-------|
| LUSTRE_STRIPE_SIZE | Lustre-specific configuration |
| LUSTRE_STRIPE_WIDTH | Lustre-specific configuration |
| POSIX_FILENOS | File descriptor count |
| tag | Performance score (target variable) |

---

## Sample Data Comparison

### AIIO Dataset
```
Full dataset: 6M+ jobs x 46 columns
Sample provided: 100 jobs (aiio_sample_100.csv)
All values: log10(x + 1) normalized

Example row (decoded approximations):
- nprocs: ~33 processes (10^1.52 - 1)
- POSIX_OPENS: ~5,050 opens (10^3.70 - 1)
- POSIX_BYTES_READ: ~153 MB (10^8.19 - 1)
- POSIX_BYTES_WRITTEN: ~667 MB (10^8.82 - 1)
```

### Polaris Samples (Converted)

**2024 Sample (Job 2093372):**
- Processes: 8
- Runtime: 42.7 seconds
- Modules: POSIX, STDIO, APMPI, HEATMAP
- I/O: Minimal (mostly metadata operations)

**2025 Sample (Job 6162074):**
- Processes: 32
- Runtime: 1,414.7 seconds (~23.6 minutes)
- Modules: POSIX, MPI-IO, STDIO, APMPI, HEATMAP
- POSIX reads: 130,678 operations
- POSIX bytes read: 40.5 GB
- POSIX bytes written: 1.1 GB

**2026 Sample (Job 6828965):**
- Processes: 1
- Runtime: 2,231.2 seconds (~37.2 minutes)
- Modules: POSIX, STDIO, APMPI, HEATMAP

---

## Use Case Comparison

| Use Case | AIIO | Polaris Collection |
|----------|:----:|:------------------:|
| ML Training (ready-to-use) | Yes | No (requires preprocessing) |
| I/O Bottleneck Prediction | Yes | Possible (with labels) |
| Raw Counter Analysis | No | Yes |
| Temporal Analysis | No | Yes (DXT traces, timestamps) |
| File-level Analysis | No | Yes |
| Cross-job Correlation | Limited | Yes (user/file hashes) |
| MPI-IO Analysis | No | Yes |
| System-wide Trends | No | Yes |

---

## Preprocessing Requirements

### To Use Polaris Data Like AIIO:
1. Parse .darshan files using darshan-parser or PyDarshan
2. Aggregate per-file counters to job-level
3. Select relevant POSIX counters (45 features)
4. Apply log10(x + 1) normalization
5. Generate labels (requires benchmark data or expert annotation)

### Sample Conversion Commands:
```bash
# Full text dump
darshan-parser --all file.darshan > output.txt

# Performance metrics
darshan-parser --perf file.darshan > perf.txt

# Aggregated totals
darshan-parser --total file.darshan > totals.txt

# Python extraction
python -c "
import darshan
report = darshan.DarshanReport('file.darshan')
df = report.records['POSIX'].to_df()
"
```

---

## Converted File Reference

Files in `sample_darshan_output/YYYY/`:

| File | Description |
|------|-------------|
| `sample_YYYY_full.txt` | Complete darshan-parser output |
| `sample_YYYY_base.txt` | Base counter data |
| `sample_YYYY_perf.txt` | Derived performance metrics |
| `sample_YYYY_total.txt` | Aggregated counter totals |
| `sample_YYYY_files.txt` | Per-file summary |
| `sample_YYYY_dxt.txt` | DXT trace data (if available) |
| `sample_YYYY_job_info.json` | Job metadata in JSON |

---

## Why Polaris Dataset is More Complete for Bottleneck Detection

The ALCF Polaris collection offers significant advantages over the tabular AIIO CSV format for comprehensive I/O performance bottleneck detection:

### 1. Multi-Module Coverage

| Bottleneck Type | AIIO Coverage | Polaris Coverage |
|-----------------|:-------------:|:----------------:|
| POSIX-level inefficiencies | Partial | Full |
| MPI-IO collective issues | None | Full |
| STDIO buffering problems | None | Full |
| Metadata overhead | Partial | Full |
| Temporal patterns | None | Full (HEATMAP) |
| Rank imbalance | None | Full |

### 2. Timing Information (Critical for Bottleneck Detection)

**AIIO Dataset**: No timing data - only operation counts and sizes.

**Polaris Dataset** includes:
- `F_READ_TIME`, `F_WRITE_TIME` - actual I/O time spent
- `F_META_TIME` - metadata operation overhead
- `F_OPEN_START_TIMESTAMP`, `F_CLOSE_END_TIMESTAMP` - temporal context
- `F_FASTEST_RANK_TIME`, `F_SLOWEST_RANK_TIME` - load imbalance detection
- `F_VARIANCE_RANK_TIME` - consistency analysis

**Why this matters**: Without timing data, you cannot distinguish between:
- A job that did 1000 reads quickly (efficient)
- A job that did 1000 reads slowly (bottleneck)

### 3. Rank-Level Statistics (Essential for Parallel I/O Analysis)

**Polaris-exclusive counters**:
```
POSIX_FASTEST_RANK / POSIX_SLOWEST_RANK
POSIX_FASTEST_RANK_BYTES / POSIX_SLOWEST_RANK_BYTES
POSIX_F_VARIANCE_RANK_BYTES / POSIX_F_VARIANCE_RANK_TIME
```

**Bottlenecks detectable**:
- Load imbalance (some ranks doing more I/O than others)
- Serialization (one rank becomes bottleneck)
- Stragglers (one slow rank delays entire job)

### 4. MPI-IO Module (Parallel I/O Pattern Analysis)

AIIO has **zero** MPI-IO counters. Polaris includes 65+ MPI-IO counters:

| Counter Category | Bottleneck Insight |
|------------------|-------------------|
| `MPIIO_COLL_READS/WRITES` | Collective vs independent I/O choice |
| `MPIIO_INDEP_READS/WRITES` | Uncoordinated access patterns |
| `MPIIO_NB_READS/WRITES` | Non-blocking I/O usage |
| `MPIIO_VIEWS` | Data type optimizations |
| `MPIIO_HINTS` | Tuning parameter usage |

### 5. Extended Size Histograms

**AIIO**: Limited size buckets (missing 10K-100K, 1M-4M, 4M-10M, etc.)

**Polaris**: Complete histogram coverage:
```
SIZE_READ/WRITE_0_100
SIZE_READ/WRITE_100_1K
SIZE_READ/WRITE_1K_10K
SIZE_READ/WRITE_10K_100K    # Missing in AIIO
SIZE_READ/WRITE_100K_1M
SIZE_READ/WRITE_1M_4M       # Missing in AIIO
SIZE_READ/WRITE_4M_10M      # Missing in AIIO
SIZE_READ/WRITE_10M_100M    # Missing in AIIO
SIZE_READ/WRITE_100M_1G     # Missing in AIIO
SIZE_READ/WRITE_1G_PLUS     # Missing in AIIO
```

**Why this matters**: Small I/O sizes often indicate inefficient access patterns. Complete histograms enable precise bottleneck categorization.

### 6. File-Level Granularity

**AIIO**: Job-level aggregation only (one row per job).

**Polaris**: Per-file records enable:
- Identifying specific files causing bottlenecks
- Detecting file system mount point issues (`/lus/eagle` vs `/home`)
- Analyzing shared vs unique file access patterns

### 7. HEATMAP Module (Temporal Distribution)

Polaris includes temporal I/O distribution data that reveals:
- I/O bursts vs sustained throughput
- Synchronization barriers
- Periodic checkpoint patterns
- End-of-job I/O storms

### 8. Bottleneck Detection Capabilities Comparison

| Bottleneck Category | AIIO | Polaris | Key Enabling Counters |
|--------------------|:----:|:-------:|----------------------|
| Small I/O operations | Partial | Full | SIZE_* histograms |
| Non-sequential access | Yes | Yes | SEQ_*, CONSEC_* |
| Poor alignment | Yes | Yes | *_NOT_ALIGNED |
| Read/write imbalance | Partial | Full | Timing + bytes |
| Metadata overhead | Limited | Full | F_META_TIME, STATS |
| Collective I/O misuse | No | Full | MPIIO_COLL_* |
| Load imbalance | No | Full | *_RANK statistics |
| Temporal patterns | No | Full | HEATMAP, timestamps |
| File contention | No | Yes | Per-file records |

### 9. Practical Impact for ML-Based Detection

To build a comprehensive I/O bottleneck detection model, the Polaris dataset enables:

1. **Multi-label classification**: Detect multiple simultaneous bottlenecks
2. **Severity estimation**: Use timing data to quantify impact
3. **Root cause analysis**: Trace issues to specific files/ranks
4. **Temporal modeling**: Learn patterns over time with HEATMAP data
5. **Transfer learning**: Pre-train on 1.37M unlabeled logs, fine-tune on benchmarks

### 10. Preprocessing Pipeline for Polaris Data

To leverage Polaris data for bottleneck detection:

```python
# Extract features similar to AIIO format
features = {
    # Basic counters (like AIIO)
    'nprocs': job['nprocs'],
    'POSIX_READS': totals['POSIX_READS'],
    'POSIX_WRITES': totals['POSIX_WRITES'],
    # ... other POSIX counters

    # NEW: Timing ratios (bottleneck indicators)
    'read_time_ratio': totals['F_READ_TIME'] / job['run_time'],
    'meta_time_ratio': totals['F_META_TIME'] / job['run_time'],

    # NEW: Efficiency metrics
    'read_bandwidth': totals['POSIX_BYTES_READ'] / totals['F_READ_TIME'],
    'write_bandwidth': totals['POSIX_BYTES_WRITTEN'] / totals['F_WRITE_TIME'],

    # NEW: Load balance indicators
    'rank_time_variance': totals['F_VARIANCE_RANK_TIME'],
    'rank_bytes_variance': totals['F_VARIANCE_RANK_BYTES'],

    # NEW: MPI-IO indicators
    'collective_ratio': totals['MPIIO_COLL_WRITES'] / (totals['MPIIO_INDEP_WRITES'] + 1),
}

# Apply log normalization (like AIIO)
features_normalized = {k: np.log10(v + 1) for k, v in features.items()}
```

---

## References

1. **AIIO Paper**: Dong, B., Bez, J.L., & Byna, S. (2023). "AIIO: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis." HPDC '23. https://doi.org/10.1145/3588195.3592986

2. **AIIO GitHub**: https://github.com/hpc-io/aiio

3. **Polaris Collection**: ALCF Polaris Darshan Log Collection (2025). https://doi.org/10.5281/zenodo.15052603

4. **Darshan Documentation**: https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html
