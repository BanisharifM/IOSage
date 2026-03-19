# data/processed/ — Pipeline Output Files

## Directory Structure

```
data/processed/
├── production/                              # 131K ALCF Polaris production logs
│   ├── raw_features.parquet                 # Stage 1: all 1.37M logs, 156 cols
│   ├── cleaned_features.parquet             # Stage 2: 131K filtered, 156 cols
│   ├── features.parquet                     # Stage 3: + 39 derived = 195 cols (TRAINING INPUT)
│   ├── features_normalized.parquet          # Stage 5: log1p + RobustScaler, 166 cols (MLP only)
│   ├── labels.parquet                       # Drishti heuristic labels: 8 dims + 30 codes
│   ├── scalers.pkl                          # RobustScaler fitted on train split only
│   ├── split_indices.pkl                    # Temporal 70/15/15 split indices
│   ├── dropped_features.json               # 29 features excluded in Stage 5
│   ├── dataset_stats.json                   # Raw dataset statistics
│   ├── splits/                              # Pre-split normalized parquets
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── eda/                                 # Exploratory Data Analysis
│       ├── stats.parquet                    # Per-feature statistics
│       ├── correlation.parquet              # Spearman correlation matrix
│       └── report.json                      # Summary report
├── benchmark/                               # Benchmark ground-truth logs
│   ├── features.parquet                     # Same pipeline as production (EVALUATION INPUT)
│   └── labels.parquet                       # Construction-based labels from benchmark configs
└── README.md                                # This file
```

## Pipeline Overview

```
Production: Darshan Logs (1.37M) -> parse -> raw (1.37M x 156)
                                      -> clean (131K x 156) -> engineer (131K x 195)
                                                                  -> normalize (131K x 166)
                                                                  -> Drishti labels (131K x 43)

Benchmark:  Darshan Logs (3,344) -> parse -> engineer (617 x 198)
                                              -> construction labels (617 x 14)

Both use: parse_darshan_log() -> extract_raw_features() -> stage3_engineer()
Verified at VALUE level: 0 differences across 186 features (2026-03-19).
```

## File Details

### Production Data (131K Polaris logs)

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `production/raw_features.parquet` | 1,397,216 | 156 | ALL logs, raw Darshan counters |
| `production/cleaned_features.parquet` | 131,151 | 156 | Filtered: require POSIX, min 1s, min 1KB |
| `production/features.parquet` | 131,151 | 195 | Cleaned + 39 derived features. **Tree model training input.** |
| `production/features_normalized.parquet` | 131,151 | 166 | log1p + RobustScaler. **MLP training input.** |
| `production/labels.parquet` | 131,151 | 43 | Drishti heuristic: 8 dimensions + 30 codes + confidence |

### Benchmark Ground-Truth (617 benchmark jobs)

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `benchmark/features.parquet` | 617 | 198 | Same pipeline as production + 3 metadata cols |
| `benchmark/labels.parquet` | 617 | 14 | Construction labels from benchmark configs |

### Scalers and Splits

| File | Description |
|------|-------------|
| `production/scalers.pkl` | RobustScaler fitted on train split ONLY |
| `production/split_indices.pkl` | Temporal split: train 70%, val 15%, test 15% |
| `production/splits/*.parquet` | Pre-split normalized features |

## Which Files Does Training Use?

### Tree Models (XGBoost, LightGBM, RF)
- Training features: `production/features.parquet` (195 cols, NOT normalized)
- Training labels: `production/labels.parquet`
- Split indices: `production/split_indices.pkl`
- GT test features: `benchmark/features.parquet`
- GT test labels: `benchmark/labels.parquet`

### Neural Models (MLP)
- Training features: `production/features_normalized.parquet` (166 cols)
- GT test features: Normalize `benchmark/features.parquet` using `production/scalers.pkl`

## All Paths Configured In

`configs/training.yaml` — single source of truth for all file paths.
Code reads paths from config, never hardcodes them.
