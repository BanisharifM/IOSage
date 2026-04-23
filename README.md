# IOSage: Benchmark-Grounded Multi-Label I/O Bottleneck Diagnosis with Validated Recommendations

Submitted to [SC 2026](https://sc26.supercomputing.org/) (Supercomputing Conference).

**Track:** Performance Measurement, Modeling, & Tools
**Format:** IEEE proceedings, 10 pages excl. bibliography, double-anonymous

## Overview

IOSage is a hybrid ML+LLM system that combines trained classifiers for multi-label I/O bottleneck detection with a retrieval-augmented LLM for code-level fix recommendations. Trained on 1.37M Darshan logs from ALCF Polaris and benchmark-verified ground truth.

### Architecture

```
Darshan Log -> Feature Extraction (157 features)
            -> ML Classifier (XGBoost/LightGBM)
            -> SHAP Attribution (top-K features)
            -> RAG from Benchmark Knowledge Base
            -> LLM Code-Level Recommendations
```

### Key Contributions

- Facility-scale I/O bottleneck study on 1.37M production Darshan logs from ALCF Polaris
- IOSage three-stage pipeline: ML detection as precision gate for LLM recommendations (0.929 Micro-F1, 94% false-positive reduction)
- DIOBench: benchmark-grounded knowledge base (689 entries) and test set (488 traces) from six benchmark suites
- Closed-loop validation with four LLMs (4.5-11.4x geometric-mean speedup) and 9 real applications from four HPC facilities

## Project Structure

```
IOSage/
├── src/                     # Source code
│   ├── data/                # Darshan parsing, feature extraction, preprocessing
│   ├── models/              # ML classifiers, training, SHAP attribution
│   ├── llm/                 # LLM recommendation, knowledge base, iterative optimizer
│   ├── ioprescriber/        # End-to-end pipeline (detector, retriever, recommender)
│   └── utils/               # Metrics, visualization, logging
├── configs/                 # Hyperparameter YAML files
├── scripts/                 # Reproduction, figure generation, verification
├── benchmarks/              # Ground truth generation (IOR, mdtest, DLIO, h5bench, HACC-IO, custom)
├── data/
│   ├── processed/           # Extracted features and preprocessed data
│   ├── knowledge_base/      # 689-entry benchmark-verified KB (JSON)
│   └── llm_cache/           # Cached LLM outputs for offline reproduction
├── models/                  # Trained model weights
├── results/                 # Experiment outputs and evaluation metrics
├── notebooks/               # Jupyter notebooks
└── Darshan_Logs/            # Raw data (1.37M logs, not in git)
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/BanisharifM/IOSage.git
cd IOSage
conda env create -f environment.yml
conda activate sc2026
python -c "import xgboost; print('OK')"

# Full reproduction (~65 min on 16 CPU cores)
bash scripts/reproduce_all.sh

# Quick reproduction (~20 min, pre-processed data + cached LLM outputs)
bash scripts/reproduce_all.sh --quick
```

## Dataset

1.37M anonymized Darshan I/O profiling logs from the ALCF Polaris supercomputer (Apr 2024 to Feb 2026).

**Source:** [Zenodo DOI: 10.5281/zenodo.15052603](https://doi.org/10.5281/zenodo.15052603)

| Stage | Rows | Features |
|-------|------|----------|
| Raw extraction | 1,397,216 | 186 |
| After cleaning | 131,151 | 186 |
| After exclusion | 131,151 | 157 |
| Train / Val / Test | 91,807 / 19,672 / 19,672 | 157 |

### ALCF Polaris

| Component | Specification |
|-----------|---------------|
| Nodes | 520 HPE Apollo 6500 Gen 10+ |
| CPU | AMD EPYC Milan 7543P (32-core) |
| GPU | 4x NVIDIA A100 per node |
| Storage | Eagle/Grand Lustre (100 PiB, 650 GiB/s) |

## Hardware Requirements

- **Training:** CPU only (16+ cores, 16 GB RAM minimum)
- **GPU:** Not required for any component
- **Storage:** 2 GB for processed data + models

## Software Versions

| Package | Version |
|---------|---------|
| Python | 3.9 |
| XGBoost | 2.1.4 |
| LightGBM | 4.5.0 |
| scikit-learn | 1.6.1 |
| SHAP | 0.49.1 |
| PyDarshan | 3.5.0 |

## References

- **AIIO** (HPDC'23): XGBoost+SHAP on Darshan -- [doi:10.1145/3588195.3592986](https://doi.org/10.1145/3588195.3592986)
- **IOAgent** (IPDPS'25): LLM+RAG for I/O diagnosis -- [doi:10.1109/IPDPS55747.2025](https://doi.org/10.1109/IPDPS55747.2025)
- **ION** (HotStorage'24): LLM in-context learning -- [doi:10.1145/3655038.3665948](https://doi.org/10.1145/3655038.3665948)
- **WisIO** (ICS'25): 800+ rule-based system -- [doi:10.1145/3721145.3725742](https://doi.org/10.1145/3721145.3725742)
- **Drishti** (PDSW'22): Heuristic thresholds -- [doi:10.1109/PDSW56643.2022.00007](https://doi.org/10.1109/PDSW56643.2022.00007)
- **Darshan** (ESPT'16): HPC I/O characterization -- [doi:10.1109/ESPT.2016.9](https://doi.org/10.1109/ESPT.2016.9)
