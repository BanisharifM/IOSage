# Processed data

This directory contains local pipeline products. Large data files are not part of the public Git history.

## Active resubmission layout

The maintained configuration is `configs/training_resubmission.yaml`.

```text
data/processed/resubmission/
  production/
    raw_features.parquet        merged extraction input to preprocessing
    features.parquet            cleaned and engineered training features
    labels.parquet              heuristic production labels
    split_indices.pkl           temporal production partitions
    preprocessing_manifest.json input and output identity
  benchmark/
    features.parquet            verified benchmark features
    labels.parquet              construction labels
    dataset_manifest.json       accepted sample identity and hashes
```

Each stage requires its declared upstream files and publishes a manifest only after validation. The model bundle stores the exact benchmark sample IDs assigned to training, validation, and final testing. Evaluation code reloads those IDs from the bundle and checks current input hashes before selecting rows.

At present, `data/processed/resubmission/production/` contains extraction chunks and file lists. The merged production table and the active benchmark dataset are still required before final training can run.

## Historical layout

`data/processed/production/` and `data/processed/benchmark/` contain data from the earlier artifact workflow. They remain for provenance and do not satisfy active resubmission gates. Maintained scripts must use explicit paths or `configs/training_resubmission.yaml` so historical files cannot be selected by an implicit fallback.
