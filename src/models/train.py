"""
Multi-label I/O bottleneck classifier training pipeline.

Trains Binary Relevance classifiers (one per bottleneck dimension) using
XGBoost, LightGBM, Random Forest, and MLP. Supports per-label imbalance
handling, Optuna hyperparameter tuning, and source-aware sample weighting.

Training data strategy (three phases):
  Phase 1: Heuristic labels only (131K production logs, Drishti rules)
  Phase 2: Heuristic + ground-truth (combined, source-weighted)
  Phase 3: Cleanlab-denoised heuristic + ground-truth

Evaluation: on held-out benchmark ground-truth test set (Tier 1).

References:
  - Grinsztajn et al. (NeurIPS 2022): trees > DL on tabular data
  - Ben-Baruch et al. (ICCV 2021): Asymmetric Loss for imbalanced multi-label
  - Snorkel (VLDB 2018): train on noisy labels, test on gold

Usage:
    python -m src.models.train --config configs/training.yaml
    python -m src.models.train --config configs/training.yaml --model xgboost
    python -m src.models.train --config configs/training.yaml --tune
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load features and labels, apply feature exclusions, return train/val arrays."""
    paths = config["paths"]

    # Load heuristic-labeled production data
    logger.info("Loading engineered features...")
    features_df = pd.read_parquet(PROJECT_DIR / paths["engineered_features"])
    labels_df = pd.read_parquet(PROJECT_DIR / paths["heuristic_labels"])

    # Align by _jobid
    labels_df = labels_df.set_index("_jobid")
    features_df = features_df.set_index("_jobid")
    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    logger.info("Aligned: %d samples", len(features_df))

    # Extract label columns
    dim_names = config["dimensions"]
    y = labels_df[dim_names].values.astype(np.float32)

    # Exclude non-feature columns
    exclude = set(config.get("exclude_features", []))
    # Also exclude any columns starting with _ or drishti_
    for col in features_df.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in features_df.columns if c not in exclude]
    X = features_df[feature_cols].values.astype(np.float32)

    logger.info("Features: %d columns (excluded %d)", len(feature_cols), len(exclude))

    # Load splits
    splits_path = PROJECT_DIR / paths["splits"]
    if splits_path.exists():
        splits = np.load(splits_path)
        train_idx = splits["train_indices"]
        val_idx = splits["val_indices"]
        test_idx = splits["test_indices"]
    else:
        logger.warning("No splits file found, using random 70/15/15 split")
        np.random.seed(config.get("seed", 42))
        n = len(X)
        perm = np.random.permutation(n)
        t1, t2 = int(0.7 * n), int(0.85 * n)
        train_idx, val_idx, test_idx = perm[:t1], perm[t1:t2], perm[t2:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    logger.info("Train: %d, Val: %d, Test (heuristic): %d",
                len(X_train), len(X_val), len(test_idx))

    # Load ground-truth test set (benchmark logs)
    gt_feat_path = PROJECT_DIR / paths["ground_truth_features"]
    gt_label_path = PROJECT_DIR / paths["ground_truth_labels"]

    X_gt_test, y_gt_test = None, None
    if gt_feat_path.exists() and gt_label_path.exists():
        gt_features = pd.read_parquet(gt_feat_path)
        gt_labels = pd.read_parquet(gt_label_path)

        # Use same feature columns, fill missing with 0
        gt_feat_cols = []
        for col in feature_cols:
            if col in gt_features.columns:
                gt_feat_cols.append(gt_features[col].values)
            else:
                gt_feat_cols.append(np.zeros(len(gt_features)))
        X_gt_test = np.column_stack(gt_feat_cols).astype(np.float32)
        y_gt_test = gt_labels[dim_names].values.astype(np.float32)

        logger.info("Ground-truth test set: %d samples", len(X_gt_test))
    else:
        logger.warning("Ground-truth test set not found")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_gt_test": X_gt_test, "y_gt_test": y_gt_test,
        "feature_cols": feature_cols,
        "dim_names": dim_names,
    }


def compute_scale_pos_weight(y_train, max_weight=100.0):
    """Compute per-label scale_pos_weight for imbalanced binary classification."""
    n_labels = y_train.shape[1]
    weights = []
    for i in range(n_labels):
        n_pos = y_train[:, i].sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            w = min(n_neg / n_pos, max_weight)
        else:
            w = 1.0
        weights.append(w)
    return weights


def train_xgboost_br(data, config):
    """Train Binary Relevance XGBoost (one model per label)."""
    from xgboost import XGBClassifier

    params = config["models"]["xgboost"]["params"].copy()
    seed = config["models"]["xgboost"].get("seed", 42)
    max_weight = config["imbalance"]["max_weight"]
    dim_names = data["dim_names"]

    spw = compute_scale_pos_weight(data["y_train"], max_weight)
    models = {}

    for i, dim in enumerate(dim_names):
        logger.info("  Training XGBoost for '%s' (scale_pos_weight=%.1f)...", dim, spw[i])
        clf = XGBClassifier(
            **params,
            scale_pos_weight=spw[i],
            random_state=seed,
            verbosity=0,
        )
        clf.fit(
            data["X_train"], data["y_train"][:, i],
            eval_set=[(data["X_val"], data["y_val"][:, i])],
            verbose=False,
        )
        models[dim] = clf

    return models


def train_lightgbm_br(data, config):
    """Train Binary Relevance LightGBM (one model per label)."""
    from lightgbm import LGBMClassifier

    params = config["models"]["lightgbm"]["params"].copy()
    seed = config["models"]["lightgbm"].get("seed", 42)
    max_weight = config["imbalance"]["max_weight"]
    dim_names = data["dim_names"]

    spw = compute_scale_pos_weight(data["y_train"], max_weight)
    models = {}

    for i, dim in enumerate(dim_names):
        logger.info("  Training LightGBM for '%s' (scale_pos_weight=%.1f)...", dim, spw[i])
        clf = LGBMClassifier(
            **params,
            scale_pos_weight=spw[i],
            random_state=seed,
        )
        clf.fit(
            data["X_train"], data["y_train"][:, i],
            eval_set=[(data["X_val"], data["y_val"][:, i])],
        )
        models[dim] = clf

    return models


def train_random_forest_br(data, config):
    """Train Binary Relevance Random Forest (one model per label)."""
    from sklearn.ensemble import RandomForestClassifier

    params = config["models"]["random_forest"]["params"].copy()
    seed = config["models"]["random_forest"].get("seed", 42)
    dim_names = data["dim_names"]
    models = {}

    for i, dim in enumerate(dim_names):
        logger.info("  Training RandomForest for '%s'...", dim)
        clf = RandomForestClassifier(**params, random_state=seed)
        clf.fit(data["X_train"], data["y_train"][:, i])
        models[dim] = clf

    return models


def evaluate_models(models, data, model_name):
    """Evaluate a set of per-label models on validation and ground-truth test sets."""
    from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

    dim_names = data["dim_names"]
    results = {"model": model_name, "per_label": {}}

    # Predict on validation set
    y_val_pred = np.zeros_like(data["y_val"])
    for i, dim in enumerate(dim_names):
        y_val_pred[:, i] = models[dim].predict(data["X_val"])

    results["val_micro_f1"] = f1_score(data["y_val"], y_val_pred, average="micro", zero_division=0)
    results["val_macro_f1"] = f1_score(data["y_val"], y_val_pred, average="macro", zero_division=0)
    results["val_hamming"] = hamming_loss(data["y_val"], y_val_pred)

    # Per-label metrics on validation
    for i, dim in enumerate(dim_names):
        results["per_label"][dim] = {
            "val_f1": f1_score(data["y_val"][:, i], y_val_pred[:, i], zero_division=0),
            "val_precision": precision_score(data["y_val"][:, i], y_val_pred[:, i], zero_division=0),
            "val_recall": recall_score(data["y_val"][:, i], y_val_pred[:, i], zero_division=0),
        }

    # Predict on ground-truth test set
    if data["X_gt_test"] is not None:
        y_gt_pred = np.zeros_like(data["y_gt_test"])
        for i, dim in enumerate(dim_names):
            y_gt_pred[:, i] = models[dim].predict(data["X_gt_test"])

        results["gt_micro_f1"] = f1_score(data["y_gt_test"], y_gt_pred, average="micro", zero_division=0)
        results["gt_macro_f1"] = f1_score(data["y_gt_test"], y_gt_pred, average="macro", zero_division=0)
        results["gt_hamming"] = hamming_loss(data["y_gt_test"], y_gt_pred)

        # Per-label on ground-truth
        for i, dim in enumerate(dim_names):
            n_pos = data["y_gt_test"][:, i].sum()
            if n_pos > 0:
                results["per_label"][dim]["gt_f1"] = f1_score(
                    data["y_gt_test"][:, i], y_gt_pred[:, i], zero_division=0
                )
                results["per_label"][dim]["gt_precision"] = precision_score(
                    data["y_gt_test"][:, i], y_gt_pred[:, i], zero_division=0
                )
                results["per_label"][dim]["gt_recall"] = recall_score(
                    data["y_gt_test"][:, i], y_gt_pred[:, i], zero_division=0
                )
            else:
                results["per_label"][dim]["gt_f1"] = None
                results["per_label"][dim]["gt_precision"] = None
                results["per_label"][dim]["gt_recall"] = None

        # Bootstrap CI on ground-truth
        bootstrap_cfg = config.get("evaluation", {}).get("bootstrap", {})
        if bootstrap_cfg:
            results["gt_ci"] = bootstrap_confidence_interval(
                data["y_gt_test"], y_gt_pred,
                n_resamples=bootstrap_cfg.get("n_resamples", 10000),
                confidence=bootstrap_cfg.get("confidence_level", 0.95),
                seed=bootstrap_cfg.get("seed", 42),
            )

    return results


def bootstrap_confidence_interval(y_true, y_pred, n_resamples=10000,
                                   confidence=0.95, seed=42):
    """Compute bootstrap 95% CI for Micro-F1 and Macro-F1."""
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    micro_scores = []
    macro_scores = []

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        micro_scores.append(f1_score(y_true[idx], y_pred[idx], average="micro", zero_division=0))
        macro_scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    alpha = (1 - confidence) / 2
    return {
        "micro_f1_mean": np.mean(micro_scores),
        "micro_f1_ci_lower": np.percentile(micro_scores, alpha * 100),
        "micro_f1_ci_upper": np.percentile(micro_scores, (1 - alpha) * 100),
        "macro_f1_mean": np.mean(macro_scores),
        "macro_f1_ci_lower": np.percentile(macro_scores, alpha * 100),
        "macro_f1_ci_upper": np.percentile(macro_scores, (1 - alpha) * 100),
    }


def log_results(results):
    """Log evaluation results in a formatted table."""
    model_name = results["model"]
    dim_names = list(results["per_label"].keys())

    logger.info("")
    logger.info("=" * 70)
    logger.info("Model: %s", model_name)
    logger.info("=" * 70)

    # Overall metrics
    logger.info("Validation:  Micro-F1=%.4f  Macro-F1=%.4f  Hamming=%.4f",
                results.get("val_micro_f1", 0), results.get("val_macro_f1", 0),
                results.get("val_hamming", 0))

    if "gt_micro_f1" in results:
        logger.info("Ground-truth: Micro-F1=%.4f  Macro-F1=%.4f  Hamming=%.4f",
                    results["gt_micro_f1"], results["gt_macro_f1"], results["gt_hamming"])

    if "gt_ci" in results:
        ci = results["gt_ci"]
        logger.info("Bootstrap CI (95%%): Micro-F1=[%.4f, %.4f]  Macro-F1=[%.4f, %.4f]",
                    ci["micro_f1_ci_lower"], ci["micro_f1_ci_upper"],
                    ci["macro_f1_ci_lower"], ci["macro_f1_ci_upper"])

    # Per-label table
    logger.info("")
    header = f"{'Dimension':<28s} {'Val-F1':>7s} {'Val-P':>7s} {'Val-R':>7s}"
    if "gt_micro_f1" in results:
        header += f" {'GT-F1':>7s} {'GT-P':>7s} {'GT-R':>7s}"
    logger.info(header)
    logger.info("-" * len(header))

    for dim in dim_names:
        m = results["per_label"][dim]
        row = f"{dim:<28s} {m['val_f1']:7.4f} {m['val_precision']:7.4f} {m['val_recall']:7.4f}"
        if m.get("gt_f1") is not None:
            row += f" {m['gt_f1']:7.4f} {m['gt_precision']:7.4f} {m['gt_recall']:7.4f}"
        elif "gt_micro_f1" in results:
            row += "     N/A     N/A     N/A"
        logger.info(row)

    logger.info("=" * 70)


def save_models(models, model_name, config):
    """Save trained models to disk."""
    model_dir = PROJECT_DIR / config["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{model_name}_br_models.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(models, f)
    logger.info("Saved %s models to %s", model_name, save_path)


# Global config reference for evaluate_models
config = {}


def main():
    global config
    parser = argparse.ArgumentParser(description="Train multi-label I/O bottleneck classifiers")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config path")
    parser.add_argument("--model", default="all", choices=["all", "xgboost", "lightgbm", "random_forest", "mlp"])
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--save", action="store_true", help="Save trained models")
    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    config = load_config(config_path)

    # Set global seed
    np.random.seed(config.get("seed", 42))

    logger.info("Loading data...")
    t0 = time.time()
    data = load_data(config)
    logger.info("Data loaded in %.1fs", time.time() - t0)

    # Log label distribution
    dim_names = data["dim_names"]
    logger.info("")
    logger.info("Training label distribution:")
    for i, dim in enumerate(dim_names):
        n_pos = data["y_train"][:, i].sum()
        pct = 100 * n_pos / len(data["y_train"])
        logger.info("  %-28s %6d (%5.1f%%)", dim, n_pos, pct)

    # Train models
    trainers = {
        "xgboost": train_xgboost_br,
        "lightgbm": train_lightgbm_br,
        "random_forest": train_random_forest_br,
    }

    models_to_train = list(trainers.keys()) if args.model == "all" else [args.model]
    all_results = []

    for model_name in models_to_train:
        model_cfg = config["models"].get(model_name, {})
        if not model_cfg.get("enabled", False):
            logger.info("Skipping %s (disabled in config)", model_name)
            continue

        logger.info("")
        logger.info("Training %s (Binary Relevance)...", model_name)
        t0 = time.time()
        models = trainers[model_name](data, config)
        train_time = time.time() - t0
        logger.info("%s training completed in %.1fs", model_name, train_time)

        # Evaluate
        results = evaluate_models(models, data, model_name)
        results["train_time"] = train_time
        log_results(results)
        all_results.append(results)

        # Save if requested
        if args.save:
            save_models(models, model_name, config)

    # Save results summary
    results_dir = PROJECT_DIR / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "training_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    logger.info("Results saved to %s", results_path)

    # Summary comparison
    if len(all_results) > 1:
        logger.info("")
        logger.info("=" * 50)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 50)
        header = f"{'Model':<20s} {'Val-MiF1':>9s} {'Val-MaF1':>9s} {'GT-MiF1':>9s} {'GT-MaF1':>9s} {'Time':>7s}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in all_results:
            row = f"{r['model']:<20s} {r['val_micro_f1']:9.4f} {r['val_macro_f1']:9.4f}"
            if "gt_micro_f1" in r:
                row += f" {r['gt_micro_f1']:9.4f} {r['gt_macro_f1']:9.4f}"
            else:
                row += "       N/A       N/A"
            row += f" {r['train_time']:6.1f}s"
            logger.info(row)
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
