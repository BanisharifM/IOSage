# IOSage notebooks

The notebooks consume current manifested runs. They do not contain stored outputs or fixed result assertions.

Install the notebook extra, then run from this directory after setting the paths named in each notebook:

```bash
conda activate iosage
python -m pip install -r requirements-notebooks.txt
cd notebooks
jupyter notebook
```

| Notebook | Required input |
|---|---|
| `01_reproduce_main_results.ipynb` | `IOSAGE_TRAINING_RUN`, an immutable final-evaluation training run |
| `02_shap_analysis.ipynb` | `IOSAGE_ATTRIBUTION_DIR`, output from `src.models.attribution` |
| `03_data_exploration.ipynb` | `IOSAGE_PRODUCTION_DIR` and `IOSAGE_BENCHMARK_DIR`, both with passed manifests |
