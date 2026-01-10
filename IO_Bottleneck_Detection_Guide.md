# I/O Performance Bottleneck Detection with Darshan Logs

## Executive Summary

Based on research from multiple papers and HPC centers, **POSIX counters alone are NOT sufficient** for complete bottleneck analysis. You need to examine multiple modules (POSIX, MPI-IO, STDIO, and optionally DXT for fine-grained tracing) along with their relationships.

---

## 1. Critical Metrics by Module

### POSIX Module - Core Metrics

| Counter | What to Look For | Bottleneck Indicator |
|---------|------------------|---------------------|
| `POSIX_BYTES_READ/WRITTEN` | Total data volume | Low values with high operation counts = small I/O problem |
| `POSIX_READS/WRITES` | Operation counts | High counts with low bytes = inefficient access |
| `POSIX_SIZE_*` histograms | Access size distribution | Many operations in 0-100 byte or 100-1K range = **critical bottleneck** |
| `POSIX_CONSEC_READS/WRITES` | Consecutive accesses | Low values = fragmented/random I/O |
| `POSIX_SEQ_READS/WRITES` | Sequential with gaps | High seq but low consecutive = strided access |
| `POSIX_FILE_NOT_ALIGNED` | Alignment issues | High values = stripe boundary misalignment |
| `POSIX_MEM_NOT_ALIGNED` | Memory alignment | Can cause extra memory copies |
| `POSIX_RW_SWITCHES` | Read/write alternation | High values = poor access pattern |
| `POSIX_STRIDE[1-4]_*` | Common stride patterns | Reveals access regularity |
| `POSIX_F_READ/WRITE_TIME` | Cumulative I/O time | Compare to `META_TIME` |
| `POSIX_F_META_TIME` | Metadata overhead | If > data time = **metadata bottleneck** |
| `POSIX_FASTEST/SLOWEST_RANK_TIME` | Load imbalance | Large difference = **I/O imbalance** |
| `POSIX_F_VARIANCE_RANK_*` | Variance across ranks | High variance = load imbalance |

### MPI-IO Module - Parallel I/O Efficiency

| Counter | What to Look For | Bottleneck Indicator |
|---------|------------------|---------------------|
| `MPIIO_INDEP_READS/WRITES` | Independent operations | High counts = missed collective optimization |
| `MPIIO_COLL_READS/WRITES` | Collective operations | Low/zero = **not using collective I/O** |
| `MPIIO_HINTS` | Hints applied | Zero = not tuning MPI-IO |
| `MPIIO_VIEWS` | File views used | Zero with parallel I/O = inefficient |
| Ratio: `MPIIO_OPS / POSIX_OPS` | Aggregation efficiency | 1:1 ratio = **no I/O aggregation occurring** |

### STDIO Module

| Counter | What to Look For | Bottleneck Indicator |
|---------|------------------|---------------------|
| `STDIO_READS/WRITES` | Operation counts | High counts with buffered I/O = potential double-buffering |
| `STDIO_BYTES_*` | Data volume | Compare to POSIX to detect library overhead |
| `STDIO_FLUSHES` | Flush frequency | Excessive flushes = performance killer |

### DXT (Extended Tracing) - Fine-Grained Analysis

When default counters aren't enough, DXT provides:
- **Per-operation timestamps** - identify slow operations
- **Offset patterns** - detect strided vs random access
- **Rank-level behavior** - find stragglers
- **I/O phases** - correlate with application phases

---

## 2. Key Bottleneck Patterns to Detect

### Pattern 1: Small I/O Operations
```
Indicator: POSIX_SIZE_WRITE_0_100 >> POSIX_SIZE_WRITE_1M_4M
Impact: Orders of magnitude performance loss
Solution: Buffer and aggregate writes
```

### Pattern 2: Metadata Dominance
```
Indicator: POSIX_F_META_TIME > POSIX_F_READ_TIME + POSIX_F_WRITE_TIME
Impact: Open/close/stat operations dominating
Solution: Reduce file count, cache handles
```

### Pattern 3: Load Imbalance
```
Indicator: POSIX_SLOWEST_RANK_TIME >> POSIX_FASTEST_RANK_TIME
         or high POSIX_F_VARIANCE_RANK_TIME
Impact: All ranks wait for slowest
Solution: Redistribute I/O, use collective operations
```

### Pattern 4: Missing Collective I/O
```
Indicator: MPIIO_INDEP_WRITES > 0 && MPIIO_COLL_WRITES == 0
Impact: No I/O aggregation
Solution: Use MPI_File_write_all instead of MPI_File_write
```

### Pattern 5: Misaligned Access
```
Indicator: High POSIX_FILE_NOT_ALIGNED values
Impact: Cross-stripe access, false sharing
Solution: Align to stripe size (typically 1MB on Lustre)
```

### Pattern 6: Random/Non-Sequential Access
```
Indicator: POSIX_SEQ_* << POSIX_READS/WRITES
         Low POSIX_CONSEC_* values
Impact: Poor prefetching, cache misses
Solution: Sort I/O requests, use collective I/O
```

### Pattern 7: Read-Write Thrashing
```
Indicator: High POSIX_RW_SWITCHES
Impact: Cache invalidation, lock contention
Solution: Separate read/write phases
```

---

## 3. Derived Metrics to Calculate

Research papers recommend calculating these derived metrics:

```python
# I/O Efficiency Metrics
avg_read_size = POSIX_BYTES_READ / POSIX_READS
avg_write_size = POSIX_BYTES_WRITTEN / POSIX_WRITES

# Bandwidth (need runtime)
read_bandwidth = POSIX_BYTES_READ / runtime
write_bandwidth = POSIX_BYTES_WRITTEN / runtime

# Metadata Overhead Ratio
metadata_ratio = POSIX_F_META_TIME / (POSIX_F_READ_TIME + POSIX_F_WRITE_TIME + POSIX_F_META_TIME)

# I/O Aggregation Ratio (MPI-IO)
aggregation_ratio = MPIIO_OPS / POSIX_OPS  # Target: < 1

# Sequential Access Ratio
seq_ratio = (POSIX_SEQ_READS + POSIX_CONSEC_READS) / POSIX_READS

# Load Imbalance Factor
imbalance = POSIX_SLOWEST_RANK_TIME / POSIX_FASTEST_RANK_TIME  # Target: ~1

# Alignment Efficiency
alignment_eff = 1 - (POSIX_FILE_NOT_ALIGNED / (POSIX_READS + POSIX_WRITES))
```

---

## 4. Recommended Analysis Workflow

Based on the research (especially UMAMI and Zoom-in Analysis):

```
1. HIGH-LEVEL TRIAGE
   └── Check total bytes, runtime, bandwidth
   └── Compare to expected/peak performance

2. ACCESS PATTERN ANALYSIS
   └── Examine size histograms (SIZE_*_* counters)
   └── Check sequential vs random (SEQ_*, CONSEC_*)
   └── Identify stride patterns (STRIDE*_*)

3. LAYER COMPARISON
   └── Compare MPI-IO to POSIX operations
   └── Check for aggregation (ratio should be < 1)
   └── Verify collective I/O usage

4. TIME BREAKDOWN
   └── Read time vs Write time vs Meta time
   └── Identify dominant component

5. PARALLELISM ANALYSIS
   └── Check rank variance
   └── Identify stragglers
   └── Examine shared vs unique file access

6. FINE-GRAINED (if needed)
   └── Enable DXT tracing
   └── Analyze per-operation patterns
   └── Correlate with application phases
```

---

## 5. What Research Says is Most Important

### HPC I/O Throughput Bottleneck Analysis (SC20)
> "No single metric predicts I/O performance universally; the most significant metrics depend on systems' architecture, configuration, workload characteristics, and health."

### Zoom-in Analysis (CCGrid 2019)
Key factors identified:
- Access size distribution
- File count per job
- I/O operation count
- Time spent in metadata vs data operations

### UMAMI (PDSW-DISCS 2017)
> Must consider both "I/O climate" (system capacity) and "I/O weather" (transient contention) for accurate diagnosis.

### I/O Bottleneck Detection and Tuning (PDSW 2021)
> DXT tracing is essential for connecting application behavior to performance - aggregate counters alone miss temporal patterns.

---

## 6. Summary: Top 10 Things to Check

| Priority | Check | Why |
|----------|-------|-----|
| 1 | **Access size distribution** | Small I/O is the #1 killer |
| 2 | **Metadata time ratio** | Often hidden bottleneck |
| 3 | **MPI-IO to POSIX ratio** | Reveals aggregation efficiency |
| 4 | **Collective I/O usage** | Easy optimization if missing |
| 5 | **Rank time variance** | Load imbalance detection |
| 6 | **File alignment** | Stripe boundary issues |
| 7 | **Sequential access ratio** | Random I/O is expensive |
| 8 | **File count** | Too many files = metadata storm |
| 9 | **Read/write switches** | Thrashing detection |
| 10 | **Bandwidth vs peak** | Overall efficiency check |

---

## 7. Complete Counter Reference

### POSIX Counters (Full List)

**Operation Counts:**
- `POSIX_OPENS` - Count of file opens (including fileno and dup)
- `POSIX_FILENOS` - Count of fileno operations
- `POSIX_DUPS` - Count of dup operations
- `POSIX_READS` - Count of read operations
- `POSIX_WRITES` - Count of write operations
- `POSIX_SEEKS` - Count of seek operations
- `POSIX_STATS` - Count of stat operations
- `POSIX_MMAPS` - Count of mmap operations
- `POSIX_FSYNCS` - Count of fsync operations
- `POSIX_FDSYNCS` - Count of fdatasync operations

**Data Volume:**
- `POSIX_BYTES_READ` - Total bytes read
- `POSIX_BYTES_WRITTEN` - Total bytes written
- `POSIX_MAX_BYTE_READ` - Highest offset read
- `POSIX_MAX_BYTE_WRITTEN` - Highest offset written

**Access Patterns:**
- `POSIX_CONSEC_READS` - Count of consecutive reads (no gaps)
- `POSIX_CONSEC_WRITES` - Count of consecutive writes (no gaps)
- `POSIX_SEQ_READS` - Count of sequential reads (with gaps)
- `POSIX_SEQ_WRITES` - Count of sequential writes (with gaps)
- `POSIX_RW_SWITCHES` - Count of read/write direction changes
- `POSIX_STRIDE1_STRIDE` through `POSIX_STRIDE4_STRIDE` - Four most common strides
- `POSIX_STRIDE1_COUNT` through `POSIX_STRIDE4_COUNT` - Counts of common strides

**Access Size Histograms:**
- `POSIX_SIZE_READ_0_100` - Reads 0-100 bytes
- `POSIX_SIZE_READ_100_1K` - Reads 100B-1KB
- `POSIX_SIZE_READ_1K_10K` - Reads 1KB-10KB
- `POSIX_SIZE_READ_10K_100K` - Reads 10KB-100KB
- `POSIX_SIZE_READ_100K_1M` - Reads 100KB-1MB
- `POSIX_SIZE_READ_1M_4M` - Reads 1MB-4MB
- `POSIX_SIZE_READ_4M_10M` - Reads 4MB-10MB
- `POSIX_SIZE_READ_10M_100M` - Reads 10MB-100MB
- `POSIX_SIZE_READ_100M_1G` - Reads 100MB-1GB
- `POSIX_SIZE_READ_1G_PLUS` - Reads > 1GB
- (Same pattern for `POSIX_SIZE_WRITE_*`)

**Alignment:**
- `POSIX_MEM_NOT_ALIGNED` - Count of unaligned memory accesses
- `POSIX_MEM_ALIGNMENT` - Memory alignment value
- `POSIX_FILE_NOT_ALIGNED` - Count of unaligned file accesses
- `POSIX_FILE_ALIGNMENT` - File alignment value (stripe size)

**Timing:**
- `POSIX_F_OPEN_START_TIMESTAMP` - First open timestamp
- `POSIX_F_READ_START_TIMESTAMP` - First read timestamp
- `POSIX_F_WRITE_START_TIMESTAMP` - First write timestamp
- `POSIX_F_CLOSE_START_TIMESTAMP` - First close timestamp
- `POSIX_F_OPEN_END_TIMESTAMP` - Last open timestamp
- `POSIX_F_READ_END_TIMESTAMP` - Last read timestamp
- `POSIX_F_WRITE_END_TIMESTAMP` - Last write timestamp
- `POSIX_F_CLOSE_END_TIMESTAMP` - Last close timestamp
- `POSIX_F_READ_TIME` - Cumulative read time
- `POSIX_F_WRITE_TIME` - Cumulative write time
- `POSIX_F_META_TIME` - Cumulative metadata time

**Parallel I/O (Shared Files):**
- `POSIX_FASTEST_RANK` - Rank with fastest I/O
- `POSIX_FASTEST_RANK_BYTES` - Bytes by fastest rank
- `POSIX_SLOWEST_RANK` - Rank with slowest I/O
- `POSIX_SLOWEST_RANK_BYTES` - Bytes by slowest rank
- `POSIX_F_FASTEST_RANK_TIME` - Time for fastest rank
- `POSIX_F_SLOWEST_RANK_TIME` - Time for slowest rank
- `POSIX_F_VARIANCE_RANK_TIME` - Variance in rank times
- `POSIX_F_VARIANCE_RANK_BYTES` - Variance in rank bytes

### MPI-IO Counters

**Operation Counts:**
- `MPIIO_INDEP_OPENS` - Independent open calls
- `MPIIO_COLL_OPENS` - Collective open calls
- `MPIIO_INDEP_READS` - Independent read calls
- `MPIIO_INDEP_WRITES` - Independent write calls
- `MPIIO_COLL_READS` - Collective read calls
- `MPIIO_COLL_WRITES` - Collective write calls
- `MPIIO_SPLIT_READS` - Split collective reads
- `MPIIO_SPLIT_WRITES` - Split collective writes
- `MPIIO_NB_READS` - Non-blocking reads
- `MPIIO_NB_WRITES` - Non-blocking writes
- `MPIIO_SYNCS` - Sync operations
- `MPIIO_HINTS` - Hints set
- `MPIIO_VIEWS` - Views set

**Data Volume:**
- `MPIIO_BYTES_READ` - Total bytes read at MPI level
- `MPIIO_BYTES_WRITTEN` - Total bytes written at MPI level

**Timing:**
- `MPIIO_F_READ_TIME` - Cumulative MPI read time
- `MPIIO_F_WRITE_TIME` - Cumulative MPI write time
- `MPIIO_F_META_TIME` - Cumulative MPI metadata time

### STDIO Counters

- `STDIO_OPENS` - fopen calls
- `STDIO_READS` - fread/fgets/etc calls
- `STDIO_WRITES` - fwrite/fputs/etc calls
- `STDIO_SEEKS` - fseek calls
- `STDIO_FLUSHES` - fflush calls
- `STDIO_BYTES_READ` - Total bytes read
- `STDIO_BYTES_WRITTEN` - Total bytes written
- `STDIO_F_READ_TIME` - Cumulative read time
- `STDIO_F_WRITE_TIME` - Cumulative write time
- `STDIO_F_META_TIME` - Cumulative metadata time

---

## 8. Tools for Analysis

### Command-Line Tools
```bash
# Basic parsing
darshan-parser <log.darshan>

# Generate PDF summary
darshan-job-summary.pl <log.darshan>

# Compare two logs
darshan-diff <log1.darshan> <log2.darshan>

# Analyze directory of logs
darshan-analyzer <directory>
```

### Python (PyDarshan)
```python
import darshan

report = darshan.DarshanReport("file.darshan")

# Access POSIX data
posix_df = report.records['POSIX'].to_df()

# Access MPI-IO data
mpiio_df = report.records['MPI-IO'].to_df()

# Get job metadata
metadata = report.metadata
```

### DXT Explorer
For fine-grained trace visualization:
- GitHub: https://github.com/hpc-io/dxt-explorer
- Enables interactive analysis of DXT traces

---

## Sources and References

### Primary Research Papers

1. **Modular HPC I/O Characterization with Darshan** (ESPT 2016)
   - Authors: Snyder, Carns, Harms, Ross, Lockwood, Wright
   - URL: https://www.mcs.anl.gov/research/projects/darshan/

2. **UMAMI: A Recipe for Generating Meaningful Metrics through Holistic I/O Performance Analysis** (PDSW-DISCS 2017)
   - Authors: Lockwood, Yoo, Byna, Wright, Snyder, Harms, Nault, Carns
   - URL: https://dl.acm.org/doi/10.1145/3149393.3149395

3. **A Zoom-in Analysis of I/O Logs to Detect Root Causes of I/O Performance Bottlenecks** (CCGrid 2019)
   - URL: https://sdm.lbl.gov/~sbyna/research/papers/2019/201905-CCGrid-ZoomIn_IO.pdf

4. **HPC I/O Throughput Bottleneck Analysis with Explainable Local Models** (SC20)
   - URL: https://dl.acm.org/doi/10.5555/3433701.3433744

5. **I/O Bottleneck Detection and Tuning: Connecting the Dots using Interactive Log Analysis** (PDSW 2021)
   - Authors: Bez et al.
   - URL: https://www.pdsw.org/pdsw21/

6. **DXT: Darshan eXtended Tracing** (CUG 2017)
   - Authors: Xu, Snyder, Kulkarni, Venkatesan, Carns, Byna, Sisneros, Chadalavada

### Documentation and Tools

- Darshan Official: https://www.mcs.anl.gov/research/projects/darshan/
- Darshan Publications: https://wordpress.cels.anl.gov/darshan/publications/
- NERSC Darshan Docs: https://docs.nersc.gov/tools/performance/darshan/
- DXT Explorer: https://github.com/hpc-io/dxt-explorer
- PyDarshan Docs: https://darshan.readthedocs.io/
- I/O Performance Analysis Guide: https://pramodkumbhar.com/2020/03/i-o-performance-analysis-with-darshan/

### Additional Resources

- NASA HECC Darshan Guide: https://www.nas.nasa.gov/hecc/support/kb/using-darshan-for-io-characterization_681.html
- HPS Darshan Seminar: https://hps.vi4io.org/_media/teaching/autumn_term_2022/nthpda_zoya_masih_darshan.pdf
