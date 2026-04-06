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
| metadata_intensity | 1 | 2.0% |
| parallelism_efficiency | 0 | 0.0% |
| access_pattern | 3 | 6.0% |
| interface_choice | 0 | 0.0% |
| file_strategy | 1 | 2.0% |
| throughput_utilization | 8 | 16.0% |
| healthy | 24 | 48.0% |

- Healthy (no bottleneck): 24
- Single bottleneck: 22
- Multi-label: 4
- Avg dimensions per bottleneck job: 1.15

## LLM Recommendation Quality

- LLM calls made: 26 (skipped 24 healthy)
- Mean groundedness: 1.000
- Median groundedness: 1.000
- Fully grounded (1.0): 26/26
- Total recommendations generated: 53
- Average recommendations per job: 2.0

## ML vs Heuristic Agreement

Note: This is NOT accuracy -- ML was partially trained on heuristic labels.
High agreement indicates consistency; disagreements may indicate ML generalization.

- Both healthy: 24
- Both bottleneck: 25
- Heuristic=healthy, ML=bottleneck: 1
- Heuristic=bottleneck, ML=healthy: 0
- Overall agreement: 98.0%

## Representative Case Studies

### Case 1: Job 6201684

- **nprocs**: 32.0
- **runtime**: 13.0436s
- **bandwidth**: 3783.5 MB/s
- **small_io_ratio**: 0.539
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **LLM groundedness**: 1.00 (3/3)
- **Diagnosis**: This HPC job exhibits two critical I/O performance problems: (1) Small transfer granularity - with avg_write_size=103KB and avg_read_size=665KB, many operations are below the 1MB threshold for efficie
  - [access_granularity] Small I/O operations (avg 103KB writes, 665KB reads) cause excessive system call overhead. The small_io_ratio of 0.54 in
  - [interface_choice] With has_mpiio=1.0 but collective_ratio=0.28, the job uses MPI-IO but primarily independent operations. Converting to co

### Case 2: Job 2057144

- **nprocs**: 256.0
- **runtime**: 12.6013s
- **bandwidth**: 23.5 MB/s
- **small_io_ratio**: 0.000
- **heuristic labels**: ['access_granularity', 'file_strategy']
- **ML detected**: ['access_granularity', 'file_strategy']
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This job exhibits two critical I/O performance issues: (1) Small access granularity - despite having an average write size of 1MB, the job is performing many small I/O operations that create excessive
  - [access_granularity] The job is performing many small I/O operations despite showing 1MB average write size, indicating internal buffering is
  - [file_strategy] With 256 processes writing to only 9 files, there's likely shared file contention. The benchmark evidence shows this pat

### Case 3: Job 3119885

- **nprocs**: 8.0
- **runtime**: 201.617s
- **bandwidth**: 479.7 MB/s
- **small_io_ratio**: 0.190
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This HPC job exhibits two critical I/O performance problems: (1) Small transfer granularity - with average write size of 91KB and read size of 100KB, both well below the 1MB threshold for efficient I/
  - [access_granularity] The job performs many small I/O operations (avg 91KB writes, 100KB reads) causing excessive syscall overhead. Small tran
  - [interface_choice] With collective_ratio of 1.0 indicating MPI-IO usage, but still suffering from small transfer sizes, switching to collec

### Case 4: Job 5590359

- **nprocs**: 4.0
- **runtime**: 23.8208s
- **bandwidth**: 41.2 MB/s
- **small_io_ratio**: 0.569
- **heuristic labels**: ['access_granularity', 'access_pattern']
- **ML detected**: ['access_granularity', 'access_pattern']
- **LLM groundedness**: 1.00 (2/2)
- **Diagnosis**: This job exhibits severe I/O inefficiency with two critical problems: (1) Small transfer sizes - average read size of 14.8KB and write size of 66.2KB are far below the 1MB threshold for efficient I/O,
  - [access_granularity] The job's average read size of 14.8KB and write size of 66.2KB are causing excessive syscall overhead. With 56.86% small
  - [access_pattern] With only 50.19% sequential reads, the job has significant random access patterns that prevent OS read-ahead and storage

### Case 5: Job 3740445

- **nprocs**: 1.0
- **runtime**: 130391.383s
- **bandwidth**: 733.6 MB/s
- **small_io_ratio**: 0.042
- **heuristic labels**: ['healthy']
- **ML detected**: ['metadata_intensity']
- **LLM groundedness**: 1.00 (1/1)
- **Diagnosis**: This job exhibits excessive metadata operations relative to data I/O, creating a metadata_intensity bottleneck. With 1,135 files and a metadata_time_ratio of 0.2214 (22.14% of runtime spent on metadat
  - [metadata_intensity] The job creates 1,135 separate files with metadata operations consuming 22.14% of total runtime. This pattern creates ex
