# Installation Guide

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd SC_2026

# Create conda environment
# On Delta (NCSA): use project directory for env
conda env create -f environment.yml --prefix /projects/bdau/envs/sc2026
conda activate /projects/bdau/envs/sc2026

# Elsewhere: default location
# conda env create -f environment.yml
# conda activate sc2026

# OR: pip install
pip install -r requirements.txt

# Verify installation
python -c "import xgboost, lightgbm, shap, cleanlab; print('OK')"
```

## Data Setup

### Production Data (131K Polaris Darshan logs)
The processed features are included in `data/processed/production/`.
Raw Darshan logs available at: DOI 10.5281/zenodo.15052603

### Benchmark Ground-Truth Data
Processed features and labels in `data/processed/benchmark/`.
Raw benchmark Darshan logs: [Zenodo restricted link during review]

### Sample Data (for quick testing)
```bash
# Copy a few sample logs for testing
ls data/sample/  # Pre-included sample Darshan logs
```

## Reproduce Paper Results

```bash
# Full reproduction (~1 hour on CPU)
bash scripts/reproduce_all.sh

# Quick mode (skip benchmark extraction, ~20 min)
bash scripts/reproduce_all.sh --quick

# Individual steps
bash scripts/reproduce_all.sh --step 6  # Phase 2 training only
```

## Hardware Requirements

- **Training**: CPU only (AMD EPYC or Intel Xeon, 16+ cores recommended)
- **Memory**: 16 GB RAM minimum (for 91K training samples)
- **Storage**: 2 GB for processed data + models
- **GPU**: Not required (tree-based models)

## Software Versions (Tested)

| Package | Version |
|---------|---------|
| Python | 3.9.x |
| XGBoost | 2.1.4 |
| LightGBM | 4.5.0 |
| scikit-learn | 1.6.1 |
| SHAP | 0.49.1 |
| Cleanlab | 2.7.1 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |

## Project Structure

```
SC_2026/
├── configs/                    # Training hyperparameters (YAML)
├── src/
│   ├── data/                   # Parsing, feature extraction, labeling
│   └── models/                 # Training, evaluation, SHAP, biquality
├── scripts/                    # Reproduction, figure generation, verification
├── data/
│   ├── processed/
│   │   ├── production/         # 131K Polaris production logs
│   │   └── benchmark/          # 623 benchmark ground-truth logs
│   ├── benchmark_logs/         # Raw Darshan logs from benchmarks
│   └── benchmark_results/      # SLURM outputs
├── models/
│   ├── phase2/                 # Biquality trained models (best)
│   └── *.pkl                   # Phase 1 models
├── results/                    # Evaluation results, noise reports
├── paper/
│   ├── figures/                # All paper figures (30 total)
│   └── main.tex                # Paper source
├── notebooks/                  # Jupyter notebooks for reviewers
├── docs/                       # Strategy, analysis, best practices
├── requirements.txt            # Pinned Python dependencies
├── environment.yml             # Conda environment
└── INSTALL.md                  # This file
```
