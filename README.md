# ALCF Polaris Darshan Log Collection - Summary

## What Is This Dataset?

This repository contains **anonymized Darshan I/O logs** captured from production jobs running on the **ALCF Polaris** supercomputer at Argonne National Laboratory. The dataset includes **1.37+ million log files** spanning from April 2024 to January 2026.

## What Is Darshan?

Darshan is an application-level **I/O characterization tool** designed for HPC (High Performance Computing) workloads. It captures:

- **Counters** - tracking I/O operations
- **Timers** - measuring I/O duration
- **Statistics** - describing file access patterns

Darshan runs transparently on production HPC systems, generating a single compressed `.darshan` log file per instrumented application.

## Dataset Structure

```
Darshan_Logs/
├── 2024/          # Logs from 2024 (April-December)
│   └── MM/DD/     # Organized by month/day
│       └── *.darshan
├── 2025/          # Logs from 2025
│   └── MM/DD/
│       └── *.darshan
└── 2026/          # Logs from 2026 (January onwards)
    └── MM/DD/
        └── *.darshan
```

Log files are named: `[jobid]-[random_value].darshan`

## Anonymization

To protect user privacy, the following fields have been anonymized (hashed consistently across all logs):

| Field | Description |
|-------|-------------|
| `uid` | User ID |
| `exe` | Command line / executable |
| `file name suffix` | File paths (mount points preserved) |

## ALCF Polaris System Specs

| Component | Specification |
|-----------|---------------|
| Nodes | 520 HPE Apollo 6500 Gen 10+ |
| CPU | AMD EPYC Milan 7543P (32-core, 2.8 GHz) |
| RAM | 512 GB DDR4 per node |
| GPU | 4x NVIDIA A100 per node |
| Network | 2x Slingshot 11 adapters |

### Storage Systems

| Mount Point | Type | Capacity | Performance |
|-------------|------|----------|-------------|
| `/lus/eagle`, `/lus/grand` | Lustre scratch | 100 PiB | 650 GiB/s |
| `/home` | Lustre home | - | Low I/O |
| `/local/scratch` | Node-local SSD (XFS) | 3.2 TiB | 6 GiB/s |

## How to Analyze the Logs

### Option 1: PyDarshan (Recommended)

```bash
pip install darshan
```

```python
import darshan

# Read a log file
report = darshan.DarshanReport("path/to/file.darshan")

# Access data as Pandas DataFrames
posix_df = report.records['POSIX'].to_df()
```

Documentation: https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/index.html

### Option 2: darshan-util (C tools)

```bash
# Parse a log file
darshan-parser file.darshan

# Generate job summary
darshan-job-summary.pl file.darshan
```

Documentation: https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html

## Important Notes

### Compression Support Required

If you see this error:
```
Error: invalid compression type.
Error: failed to initialize decompression data structures.
```

You need bzip2 support:
- **Source install**: `apt-get install libbz2-dev` before building
- **Spack install**: `spack install darshan-util+bzip2`

### Coverage Limitations

These logs do **not** cover all Polaris jobs:
- Non-MPI applications are typically not instrumented
- Apps that don't call `MPI_Finalize()` may be missing
- Some applications explicitly disable Darshan

## Citation

If using this data, please cite:

1. **Zenodo Record**: ALCF Polaris Darshan Log Collection. (2025). doi:10.5281/zenodo.15052603

2. **Publication**: Shane Snyder, Philip Carns, Kevin Harms, Rob Latham, and Robert Ross. "Expanding Community Access to Real-world HPC Application I/O Characterization Data using Darshan". CUG 2025.

### Required Acknowledgment

> This research used resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (DOE) Office of Science user facility at Argonne National Laboratory and is based on research supported by the U.S. DOE Office of Science-Advanced Scientific Computing Research Program, under Contract No. DE-AC02-06CH11357.

## References

- [Darshan Project](https://www.mcs.anl.gov/research/projects/darshan/)
- [PyDarshan Docs](https://www.mcs.anl.gov/research/projects/darshan/docs/pydarshan/index.html)
- [Darshan Counter Reference](https://www.mcs.anl.gov/research/projects/darshan/docs/darshan-util.html#_guide_to_darshan_parser_output)
- [ALCF Public Data Repository](https://reports.alcf.anl.gov/data/index.html)
