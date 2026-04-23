# Notebooks — IOSage

Jupyter notebooks for reviewers and artifact evaluation.

## Quick Start

```bash
conda activate sc2026
cd notebooks
jupyter notebook
```

## Notebooks

| # | Notebook | Description | Runtime |
|---|----------|-------------|---------|
| 01 | `01_reproduce_main_results.ipynb` | Reproduce Table 2 (main results), all baselines | < 2 min |
| 02 | `02_shap_analysis.ipynb` | Feature attribution, beeswarm plots, domain validation | < 1 min |
| 03 | `03_data_exploration.ipynb` | Dataset overview, feature distributions, label analysis | < 1 min |

## Notes

- Notebooks load pre-trained models and pre-computed results (no training needed)
- "Full path" training can be run via `scripts/reproduce_all.sh`
- All notebooks assume the working directory is `notebooks/` and project root is `../`
