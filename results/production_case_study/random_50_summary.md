# Production Case Study: 50 Random Polaris Logs

## Overview

- **Sample size**: 50 logs (stratified: 25 bottleneck + 25 healthy)
- **Random seed**: 42
- **Source**: 131,151 production Polaris Darshan logs
- **LLM**: Claude Sonnet via OpenRouter (temperature=0)
- **Purpose**: Unbiased accuracy estimate (no cherry-picking)

## ML Detection Distribution

| Bottleneck Dimension | Count | Percentage |
|---------------------|-------|------------|
| access_granularity | 17 | 34.0% |
| metadata_intensity | 0 | 0.0% |
| parallelism_efficiency | 0 | 0.0% |
| access_pattern | 3 | 6.0% |
| interface_choice | 0 | 0.0% |
| file_strategy | 1 | 2.0% |
| throughput_utilization | 8 | 16.0% |
| healthy | 25 | 50.0% |

- Healthy (no bottleneck): 25
- Single bottleneck: 21
- Multi-label: 4
- Avg dimensions per bottleneck job: 1.16

## LLM Recommendation Quality

- LLM calls made: 25 (skipped 25 healthy)
- Mean groundedness: 1.000
- Median groundedness: 1.000
- Fully grounded (1.0): 25/25
- Total recommendations generated: 59
- Average recommendations per job: 2.4

## ML vs Heuristic Agreement

Note: This is NOT accuracy -- ML was partially trained on heuristic labels.
High agreement indicates consistency; disagreements may indicate ML generalization.

- Both healthy: 25
- Both bottleneck: 25
- Heuristic=healthy, ML=bottleneck: 0
- Heuristic=bottleneck, ML=healthy: 0
- Overall agreement: 100.0%

## Representative Case Studies

### Case 1: Job 6201684

- **nprocs**: 32.0
- **runtime**: 13.0436s
- **bandwidth**: 3783.5 MB/s
- **small_io_ratio**: 0.539
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **Top SHAP features**:
  - access_granularity: POSIX_FILE_NOT_ALIGNED (2.220), POSIX_ACCESS1_COUNT (1.747)
  - access_pattern: POSIX_FILE_NOT_ALIGNED (2.027), POSIX_WRITES (1.669)
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This HPC job exhibits two critical I/O performance problems: (1) Small transfer granularity - with 5,597 unaligned file accesses and high syscall counts (1,252 ACCESS1 operations), the application is 
  - [access_granularity] The job performs 5,438 writes and 3,800 reads with many small transfers (avg_write_size=103KB, small_io_ratio=0.5392). T
  - [access_pattern] With 5,597 unaligned file accesses and seq_write_ratio of only 0.7111, the application has fragmented access patterns th

### Case 2: Job 2057144

- **nprocs**: 256.0
- **runtime**: 12.6013s
- **bandwidth**: 23.5 MB/s
- **small_io_ratio**: 0.000
- **heuristic labels**: ['access_granularity', 'file_strategy']
- **ML detected**: ['access_granularity', 'file_strategy']
- **Top SHAP features**:
  - access_granularity: POSIX_ACCESS1_COUNT (3.161), POSIX_WRITES (3.095)
  - file_strategy: nprocs (2.468), POSIX_F_WRITE_TIME (1.215)
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This job exhibits severe I/O performance problems with 16,128 small write operations (100K-1MB range) across 256 processes writing to only 9 files. The primary issues are: (1) Excessive syscall overhe
  - [access_granularity] The job performs 16,128 write operations with many in the 100K-1MB range, causing excessive syscall overhead. Despite an
  - [file_strategy] 256 processes writing to only 9 files creates severe contention and serialization. With 719.9 seconds of write time spre

### Case 3: Job 3119885

- **nprocs**: 8.0
- **runtime**: 201.617s
- **bandwidth**: 479.7 MB/s
- **small_io_ratio**: 0.190
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **Top SHAP features**:
  - access_granularity: POSIX_WRITES (5.345), POSIX_ACCESS1_COUNT (1.562)
  - access_pattern: seq_write_ratio (2.300), seq_read_ratio (1.214)
- **LLM groundedness**: 1.00 (3/3)
- **Diagnosis**: This HPC job exhibits severe I/O inefficiencies with 1,994 POSIX write operations averaging only 91KB per write, well below the 1MB threshold for efficient I/O. The job shows poor sequential access pa
  - [access_granularity] The job performs 1,994 POSIX writes with an average size of only 91KB, causing excessive syscall overhead. Small writes 
  - [access_pattern] Poor sequential access ratios (22% write, 23% read) across 3,029 files indicate random access patterns that defeat OS re

### Case 4: Job 5590359

- **nprocs**: 4.0
- **runtime**: 23.8208s
- **bandwidth**: 41.2 MB/s
- **small_io_ratio**: 0.569
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **Top SHAP features**:
  - access_granularity: POSIX_ACCESS1_COUNT (2.429), POSIX_READS (2.041)
  - access_pattern: seq_read_ratio (2.136), POSIX_FILE_NOT_ALIGNED (1.669)
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This HPC job exhibits severe I/O inefficiency with 3,726 small POSIX reads averaging only 14.8KB per operation and 1,866 unaligned file accesses. The application performs excessive small I/O operation
  - [access_granularity] The job performs 3,726 small reads averaging 14.8KB each, far below the 1MB threshold for efficient I/O. This creates ex
  - [access_pattern] With seq_read_ratio of only 0.5019 and 1,866 unaligned accesses, the application exhibits random access patterns that de

### Case 5: Job 6882440

- **nprocs**: 32.0
- **runtime**: 210.926s
- **bandwidth**: 399.5 MB/s
- **small_io_ratio**: 0.739
- **heuristic labels**: ['access_granularity']
- **ML detected**: ['access_granularity']
- **Top SHAP features**:
  - access_granularity: POSIX_FILE_NOT_ALIGNED (2.329), POSIX_ACCESS1_COUNT (1.855)
- **LLM groundedness**: 1.00 (3/3)
- **Diagnosis**: This HPC job exhibits severe small I/O access patterns that are causing significant performance degradation. With 92,955 POSIX reads and 4,069 POSIX writes, the job is performing 97,024 total I/O oper
  - [access_granularity] The job performs 97,024 I/O operations with average sizes of 146KB (reads) and 266KB (writes), both well below the 1MB e
  - [access_granularity] With has_mpiio=1.0 indicating MPI-IO availability and collective_ratio=0.3897 showing underutilized collective operation
