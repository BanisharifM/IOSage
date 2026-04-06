"""
Compute all missing ML metrics for SC 2026 paper.

Metrics computed:
1. Micro-F1, Macro-F1 for all models
2. Hamming loss, Subset accuracy
3. Per-dimension precision, recall, F1
4. Paired bootstrap tests: XGBoost vs {LightGBM, RF, Drishti, WisIO}
5. Majority class baseline
6. Threshold-based baseline (90th percentile)
7. Bootstrap 95% CI for Micro-F1 and Macro-F1

Usage:
    /projects/bdau/envs/sc2026/bin/python scripts/compute_final_metrics.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT = Path("/work/hdd/bdau/mbanisharifdehkordi/SC_2026")
BOOST_DIR = PROJECT / "results" / "boost_experiment"
sys.path.insert(0, str(PROJECT))

DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
    "healthy",
]

SEED = 42
N_BOOTSTRAP = 10000


def load_test_data():
    """Load test features and labels from BOOST EXPERIMENT new splits."""
    test_feat = pd.read_parquet(BOOST_DIR / "new_splits/test_features.parquet")
    test_labels = pd.read_parquet(BOOST_DIR / "new_splits/test_labels.parquet")

    # Get feature columns from production config (same as training pipeline)
    import yaml
    config = yaml.safe_load(open(PROJECT / "configs/training.yaml"))
    prod_features = pd.read_parquet(PROJECT / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_features.columns if c not in exclude]

    # Build y_train from production training labels (for majority baseline)
    prod_labels = pd.read_parquet(PROJECT / config["paths"]["production_labels"])
    prod_labels = prod_labels.set_index("_jobid")
    prod_features = prod_features.set_index("_jobid")
    common = prod_features.index.intersection(prod_labels.index)
    prod_labels = prod_labels.loc[common]
    with open(PROJECT / config["paths"]["production_splits"], "rb") as f:
        prod_splits = pickle.load(f)
    train_idx = prod_splits.get("train_idx", prod_splits.get("train_indices"))
    y_train = prod_labels.iloc[train_idx][DIMS].values.astype(np.float32)
    del prod_features, prod_labels

    # Align test features to training columns
    X_cols = []
    for col in feature_cols:
        if col in test_feat.columns:
            X_cols.append(test_feat[col].values)
        else:
            X_cols.append(np.zeros(len(test_feat)))
    X_test = np.column_stack(X_cols).astype(np.float32)
    y_test = test_labels[DIMS].values.astype(np.float32)

    logger.info("Test set: %d samples, %d features, %d labels", *X_test.shape, y_test.shape[1])
    return X_test, y_test, feature_cols, y_train, test_feat, test_labels


def load_models():
    """Load XGBoost models from BOOST EXPERIMENT (only XGBoost retrained)."""
    models = {}
    # Load new XGBoost model (seed 42)
    path = BOOST_DIR / "new_models/xgboost_biquality_w100_seed42.pkl"
    with open(path, "rb") as f:
        models["xgboost"] = pickle.load(f)
    logger.info("Loaded xgboost from boost experiment (%d label models)", len(models["xgboost"]))
    return models


def predict_with_models(model_dict, X):
    """Predict using a dict of per-label models."""
    y_pred = np.zeros((len(X), len(DIMS)), dtype=np.float32)
    for i, dim in enumerate(DIMS):
        if dim in model_dict:
            y_pred[:, i] = model_dict[dim].predict(X)
    return y_pred


def compute_all_metrics(y_true, y_pred):
    """Compute comprehensive multi-label metrics."""
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(
            accuracy_score(
                [tuple(row) for row in y_true.astype(int)],
                [tuple(row) for row in y_pred.astype(int)],
            )
        ),
    }

    # Per-label metrics
    per_label = {}
    for i, dim in enumerate(DIMS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per_label[dim] = {
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
        }
    metrics["per_label"] = per_label
    return metrics


def bootstrap_ci(y_true, y_pred, n_resamples=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for Micro-F1 and Macro-F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    micro_scores, macro_scores = [], []

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        micro_scores.append(
            f1_score(y_true[idx], y_pred[idx], average="micro", zero_division=0)
        )
        macro_scores.append(
            f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        )

    return {
        "micro_f1_mean": float(np.mean(micro_scores)),
        "micro_f1_ci_lo": float(np.percentile(micro_scores, 2.5)),
        "micro_f1_ci_hi": float(np.percentile(micro_scores, 97.5)),
        "micro_f1_std": float(np.std(micro_scores)),
        "macro_f1_mean": float(np.mean(macro_scores)),
        "macro_f1_ci_lo": float(np.percentile(macro_scores, 2.5)),
        "macro_f1_ci_hi": float(np.percentile(macro_scores, 97.5)),
        "macro_f1_std": float(np.std(macro_scores)),
    }


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, name_a, name_b,
                           n_resamples=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap test: is model A significantly better than model B?

    Returns p-value (one-sided: fraction of resamples where B >= A),
    95% CI for the difference, and Cohen's d effect size.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        f1_a = f1_score(y_true[idx], y_pred_a[idx], average="micro", zero_division=0)
        f1_b = f1_score(y_true[idx], y_pred_b[idx], average="micro", zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    p_value = float(np.mean(diffs <= 0))
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # Cohen's d: mean difference / pooled std
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "cohens_d": float(cohens_d),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def majority_baseline(y_train, n_test):
    """Predict majority class per label from training distribution."""
    y_pred = np.zeros((n_test, len(DIMS)), dtype=np.float32)
    for i in range(y_train.shape[1]):
        majority = 1.0 if y_train[:, i].mean() > 0.5 else 0.0
        y_pred[:, i] = majority
    return y_pred


def threshold_baseline(X_train, X_test, feature_cols):
    """Flag if relevant feature > 90th percentile of training data.

    Uses domain-mapped features per dimension.
    """
    thresholds_90 = np.percentile(X_train, 90, axis=0)

    dim_feature_map = {
        "access_granularity": ["small_io_ratio", "small_read_ratio", "small_write_ratio"],
        "metadata_intensity": ["metadata_time_ratio", "opens_per_op", "stats_per_op"],
        "parallelism_efficiency": ["byte_imbalance", "time_imbalance"],
        "access_pattern": [],  # inverted: low seq = random
        "interface_choice": [],  # inverted: low collective = bad
        "file_strategy": ["num_files"],
        "throughput_utilization": ["fsync_ratio"],
        "healthy": [],
    }

    # Inverted features: flag if BELOW 10th percentile
    dim_inverted_map = {
        "access_pattern": ["seq_read_ratio", "seq_write_ratio"],
        "interface_choice": ["collective_ratio"],
        "throughput_utilization": ["total_bw_mb_s"],
    }

    thresholds_10 = np.percentile(X_train, 10, axis=0)

    y_pred = np.zeros((len(X_test), len(DIMS)), dtype=np.float32)

    for i, dim in enumerate(DIMS):
        # High-threshold features
        for fname in dim_feature_map.get(dim, []):
            if fname in feature_cols:
                idx = feature_cols.index(fname)
                y_pred[:, i] = np.maximum(
                    y_pred[:, i],
                    (X_test[:, idx] > thresholds_90[idx]).astype(np.float32),
                )

        # Inverted features (low = bad)
        for fname in dim_inverted_map.get(dim, []):
            if fname in feature_cols:
                idx = feature_cols.index(fname)
                y_pred[:, i] = np.maximum(
                    y_pred[:, i],
                    (X_test[:, idx] < thresholds_10[idx]).astype(np.float32),
                )

    return y_pred


def get_drishti_predictions(test_feat, test_labels):
    """Get Drishti baseline predictions on test set."""
    from src.data.drishti_labeling import compute_drishti_codes, codes_to_labels

    codes = compute_drishti_codes(test_feat)
    # codes_to_labels expects a dict of code->bool Series, returns label df
    labels_df = codes_to_labels(codes)
    y_pred = labels_df[DIMS].values.astype(np.float32)
    return y_pred


def get_wisio_predictions(test_labels):
    """Get WisIO predictions aligned to 436 test set.

    WisIO was evaluated on the full benchmark set (623 samples).
    We filter to the 436-sample test split and align by job_id + scenario.

    Returns
    -------
    y_pred : ndarray of shape (n_matched, 8)
    test_indices : ndarray of int, indices into test_labels that matched
    """
    wisio_df = pd.read_parquet(
        PROJECT / "results/wisio_baseline/wisio_predictions.parquet"
    )
    pred_cols = [f"pred_{d}" for d in DIMS]
    gt_cols = [f"gt_{d}" for d in DIMS]

    # Use job_id + scenario for unique matching (handles duplicate job_ids)
    test_df = test_labels.copy()
    test_df["_test_idx"] = range(len(test_df))

    # Check if scenario column exists in both
    merge_keys = ["job_id"]
    if "scenario" in test_df.columns and "scenario" in wisio_df.columns:
        merge_keys.append("scenario")

    # Add dup counter for remaining duplicates
    test_df["_dup"] = test_df.groupby(merge_keys).cumcount()
    wisio_df["_dup"] = wisio_df.groupby(merge_keys).cumcount()

    merged = test_df.merge(
        wisio_df[merge_keys + ["_dup"] + pred_cols],
        on=merge_keys + ["_dup"],
        how="inner",
    ).sort_values("_test_idx")

    logger.info("WisIO alignment: %d / %d test samples matched", len(merged), len(test_labels))

    test_indices = merged["_test_idx"].values
    y_pred = merged[pred_cols].values.astype(np.float32)
    return y_pred, test_indices


def print_report(name, metrics, ci=None):
    """Print formatted evaluation report."""
    logger.info("")
    logger.info("=" * 75)
    logger.info("  %s", name)
    logger.info("=" * 75)
    logger.info(
        "  Micro-F1: %.4f | Macro-F1: %.4f | Hamming: %.4f | SubsetAcc: %.4f",
        metrics["micro_f1"],
        metrics["macro_f1"],
        metrics["hamming_loss"],
        metrics["subset_accuracy"],
    )
    if ci:
        logger.info(
            "  Bootstrap 95%% CI: Micro-F1=[%.4f, %.4f]  Macro-F1=[%.4f, %.4f]",
            ci["micro_f1_ci_lo"],
            ci["micro_f1_ci_hi"],
            ci["macro_f1_ci_lo"],
            ci["macro_f1_ci_hi"],
        )
    logger.info("")
    header = f"  {'Dimension':<28s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Supp':>6s}"
    logger.info(header)
    logger.info("  " + "-" * 55)
    for dim, m in metrics["per_label"].items():
        logger.info(
            f"  {dim:<28s} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['support']:6d}"
        )
    logger.info("=" * 75)


def main():
    logger.info("Loading test data...")
    X_test, y_test, feature_cols, y_train, test_feat, test_labels = load_test_data()

    logger.info("Loading models...")
    models = load_models()

    # Training features for threshold baseline — use production training features
    import yaml
    config = yaml.safe_load(open(PROJECT / "configs/training.yaml"))
    prod_features = pd.read_parquet(PROJECT / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    prod_feature_cols = [c for c in prod_features.columns if c not in exclude]
    with open(PROJECT / config["paths"]["production_splits"], "rb") as f:
        prod_splits = pickle.load(f)
    train_idx = prod_splits.get("train_idx", prod_splits.get("train_indices"))
    X_train = prod_features.iloc[train_idx][prod_feature_cols].values.astype(np.float32)
    del prod_features

    results = {}

    # ----------------------------------------------------------------
    # 1. ML model metrics (XGBoost, LightGBM, RF)
    # ----------------------------------------------------------------
    predictions = {}
    for model_name, model_dict in models.items():
        logger.info("Evaluating %s...", model_name)
        y_pred = predict_with_models(model_dict, X_test)
        predictions[model_name] = y_pred
        metrics = compute_all_metrics(y_test, y_pred)
        ci = bootstrap_ci(y_test, y_pred)
        results[model_name] = {"metrics": metrics, "bootstrap_ci": ci}
        print_report(model_name.upper(), metrics, ci)

    # ----------------------------------------------------------------
    # 2. Drishti baseline
    # ----------------------------------------------------------------
    logger.info("Computing Drishti baseline...")
    try:
        y_pred_drishti = get_drishti_predictions(test_feat, test_labels)
        predictions["drishti"] = y_pred_drishti
        metrics_drishti = compute_all_metrics(y_test, y_pred_drishti)
        ci_drishti = bootstrap_ci(y_test, y_pred_drishti)
        results["drishti"] = {"metrics": metrics_drishti, "bootstrap_ci": ci_drishti}
        print_report("DRISHTI BASELINE", metrics_drishti, ci_drishti)
    except Exception as e:
        logger.warning("Drishti baseline failed: %s", e)
        predictions["drishti"] = None

    # ----------------------------------------------------------------
    # 3. WisIO baseline
    # ----------------------------------------------------------------
    logger.info("Computing WisIO baseline...")
    wisio_test_indices = None
    try:
        y_pred_wisio, wisio_test_indices = get_wisio_predictions(test_labels)
        y_true_wisio = y_test[wisio_test_indices]
        predictions["wisio"] = y_pred_wisio
        predictions["wisio_indices"] = wisio_test_indices
        metrics_wisio = compute_all_metrics(y_true_wisio, y_pred_wisio)
        ci_wisio = bootstrap_ci(y_true_wisio, y_pred_wisio)
        results["wisio"] = {
            "metrics": metrics_wisio,
            "bootstrap_ci": ci_wisio,
            "n_samples_evaluated": len(y_pred_wisio),
        }
        print_report("WISIO BASELINE", metrics_wisio, ci_wisio)
    except Exception as e:
        logger.warning("WisIO baseline failed: %s", e)
        import traceback
        traceback.print_exc()
        predictions["wisio"] = None

    # ----------------------------------------------------------------
    # 4. Majority class baseline
    # ----------------------------------------------------------------
    logger.info("Computing majority class baseline...")
    y_pred_majority = majority_baseline(y_train, len(y_test))
    predictions["majority"] = y_pred_majority
    metrics_majority = compute_all_metrics(y_test, y_pred_majority)
    results["majority_class"] = {"metrics": metrics_majority}
    print_report("MAJORITY CLASS BASELINE", metrics_majority)

    # ----------------------------------------------------------------
    # 5. Threshold baseline (90th percentile)
    # ----------------------------------------------------------------
    logger.info("Computing threshold baseline (90th percentile)...")
    y_pred_thresh = threshold_baseline(X_train, X_test, feature_cols)
    predictions["threshold"] = y_pred_thresh
    metrics_thresh = compute_all_metrics(y_test, y_pred_thresh)
    results["threshold_90pct"] = {"metrics": metrics_thresh}
    print_report("THRESHOLD BASELINE (P90)", metrics_thresh)

    # ----------------------------------------------------------------
    # 6. Paired bootstrap tests: XGBoost vs others
    # ----------------------------------------------------------------
    logger.info("Running paired bootstrap tests (10,000 resamples each)...")
    comparisons = {}
    y_pred_xgb = predictions["xgboost"]

    for other_name in ["drishti", "wisio"]:
        y_pred_other = predictions.get(other_name)
        if y_pred_other is None:
            logger.warning("Skipping %s comparison (no predictions)", other_name)
            continue

        # WisIO may cover a subset of test samples
        if other_name == "wisio" and wisio_test_indices is not None:
            y_true_for_comparison = y_test[wisio_test_indices]
            y_pred_xgb_for_comp = y_pred_xgb[wisio_test_indices]
            logger.info("  WisIO comparison on %d aligned samples", len(wisio_test_indices))
        else:
            y_true_for_comparison = y_test
            y_pred_xgb_for_comp = y_pred_xgb

        logger.info("  XGBoost vs %s...", other_name)
        test_result = paired_bootstrap_test(
            y_true_for_comparison, y_pred_xgb_for_comp, y_pred_other, "xgboost", other_name
        )
        comparisons[f"xgboost_vs_{other_name}"] = test_result
        logger.info(
            "    diff=%.4f [%.4f, %.4f], p=%.4f, d=%.2f, sig=%s",
            test_result["mean_diff"],
            test_result["ci_lo"],
            test_result["ci_hi"],
            test_result["p_value"],
            test_result["cohens_d"],
            test_result["significant_at_005"],
        )

    results["paired_bootstrap_tests"] = comparisons

    # ----------------------------------------------------------------
    # 7. Summary table
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("  SUMMARY TABLE")
    logger.info("=" * 80)
    logger.info(
        "  %-20s %8s %8s %8s %8s",
        "Model", "Micro-F1", "Macro-F1", "Hamming", "SubsetAcc",
    )
    logger.info("  " + "-" * 60)
    for name in ["xgboost", "drishti", "wisio",
                  "majority_class", "threshold_90pct"]:
        if name in results:
            m = results[name]["metrics"]
            logger.info(
                "  %-20s %8.4f %8.4f %8.4f %8.4f",
                name, m["micro_f1"], m["macro_f1"], m["hamming_loss"], m["subset_accuracy"],
            )
    logger.info("=" * 80)

    # ----------------------------------------------------------------
    # Save all results
    # ----------------------------------------------------------------
    output_path = BOOST_DIR / "full_evaluation/final_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
