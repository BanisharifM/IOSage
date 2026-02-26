# SC 2026: ML+LLM Hybrid for HPC I/O Bottleneck Detection

Research project for [SC 2026](https://sc26.supercomputing.org/) (Supercomputing Conference).

**Track:** Performance Measurement, Modeling, & Tools
**Format:** IEEE proceedings, 10 pages excl. bibliography, double-anonymous

## Overview

A hybrid ML+LLM system that combines trained classifiers for multi-label I/O bottleneck detection with a retrieval-augmented LLM for code-level fix recommendations. Trained on 1.37M Darshan logs from ALCF Polaris and benchmark-verified ground truth.

### Architecture

```
Darshan Log -> Feature Extraction (157 features)
            -> ML Classifier (XGBoost/LightGBM)
            -> SHAP Attribution (top-K features)
            -> RAG from Benchmark Knowledge Base
            -> LLM Code-Level Recommendations
```

### Key Contributions

- Hybrid ML+LLM architecture combining learned detection with grounded recommendations
- Benchmark-grounded Knowledge Base mapping I/O signatures to verified code fixes
- Facility-scale analysis on 1.37M production Darshan logs from ALCF Polaris
- Comprehensive evaluation against AIIO, Drishti, IOAgent baselines

## Project Structure

```
SC_2026/
├── src/                     # Source code
│   ├── data/                # Darshan parsing, feature extraction, preprocessing
│   ├── models/              # ML classifiers, training
│   └── utils/               # Metrics, visualization, logging
├── configs/                 # Hyperparameter YAML files
├── scripts/                 # Pipeline scripts (extraction, preprocessing, figures)
├── paper/                   # LaTeX paper (synced to Overleaf via SC_2026_Paper repo)
│   ├── main.tex             # Main document
│   ├── sections/            # Per-section .tex files
│   ├── figures/             # Paper figures (PDF + PNG)
│   ├── tables/              # LaTeX tables
│   └── references.bib       # Bibliography
├── benchmarks/              # Ground truth generation (IOR, mdtest, DLIO)
├── data/
│   ├── raw/ -> Darshan_Logs # Symlink to raw Polaris logs
│   ├── processed/           # Extracted features and preprocessed data (Git LFS)
│   ├── sample_logs/         # Sample Darshan parser outputs
│   └── splits/              # Train/val/test splits
├── docs/                    # Research documentation and domain knowledge
├── archive/                 # Old drafts and notes
└── Darshan_Logs/            # Raw data (1.37M logs, not in git)
```

## Setup

### Environment

```bash
# On Delta/Polaris: use the sc2026 conda environment
source activate /projects/bdau/envs/sc2026

# Or create a new environment
conda create -n sc2026 python=3.9
conda activate sc2026
pip install -r requirements.txt
```

### Git LFS

Processed parquet files are tracked with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

### Data Pipeline

```bash
# 1. Extract features from Darshan logs
python -m src.data.batch_extract --input-dir Darshan_Logs/ --output data/processed/raw_features.parquet

# 2. Run preprocessing pipeline (cleaning, engineering, normalization)
python scripts/run_preprocessing.py --config configs/preprocessing.yaml

# 3. Generate paper figures
python scripts/generate_paper_figures.py --output-dir paper/figures/preprocessing
```

### Paper

```bash
# Compile LaTeX paper
bash scripts/compile_paper.sh

# Sync to Overleaf (via SC_2026_Paper GitHub repo)
bash scripts/sync_paper_to_overleaf.sh "Update message"
```

## Dataset

1.37M anonymized Darshan I/O profiling logs from the ALCF Polaris supercomputer (Apr 2024 -- Feb 2026).

**Source:** [Zenodo DOI: 10.5281/zenodo.15052603](https://doi.org/10.5281/zenodo.15052603)

| Stage | Rows | Features |
|-------|------|----------|
| Raw extraction | 1,397,293 | 186 |
| After cleaning | 131,151 | 186 |
| After exclusion | 131,151 | 157 |
| Train / Val / Test | 91,807 / 19,672 / 19,672 | 157 |

### ALCF Polaris

| Component | Specification |
|-----------|---------------|
| Nodes | 560 HPE Apollo 6500 Gen 10+ |
| CPU | AMD EPYC Milan 7543P (32-core) |
| GPU | 4x NVIDIA A100 per node |
| Storage | Eagle/Grand Lustre (100 PiB, 650 GiB/s) |

## Key Dates

| Date | Milestone |
|------|-----------|
| March 1 | Submissions open |
| April 1 | Abstract registration |
| April 8 | Full paper submission |
| April 24 | AD appendix due |
| June 8-11 | Rebuttal period |

## References

- **AIIO** (HPDC'23): XGBoost+SHAP on Darshan -- [doi:10.1145/3588195.3592986](https://doi.org/10.1145/3588195.3592986)
- **IOAgent** (IPDPS'25): LLM+RAG for I/O diagnosis -- [doi:10.1109/IPDPS55747.2025](https://doi.org/10.1109/IPDPS55747.2025)
- **ION** (HotStorage'24): LLM in-context learning -- [doi:10.1145/3655038.3665948](https://doi.org/10.1145/3655038.3665948)
- **WisIO** (ICS'25): 800+ rule-based system -- [doi:10.1145/3721145.3725742](https://doi.org/10.1145/3721145.3725742)
- **Drishti** (PDSW'22): Heuristic thresholds -- [doi:10.1109/PDSW56643.2022.00007](https://doi.org/10.1109/PDSW56643.2022.00007)
- **Darshan** (ESPT'16): HPC I/O characterization -- [doi:10.1109/ESPT.2016.9](https://doi.org/10.1109/ESPT.2016.9)
