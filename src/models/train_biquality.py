"""
Phase 2: Biquality Learning — Weighted Combination Training

Trains on heuristic (noisy, large) + benchmark (clean, small) data together
using importance-weighted sample reweighting. This is the correct approach
for tree models (XGBoost/LightGBM) with dual label sources.

Framework: Biquality Learning (Nodet et al., Machine Learning 2023)
- One trusted dataset (benchmark, clean) + one untrusted (heuristic, noisy)
- Same feature space, same label set
- Sample weights: clean samples get W times more gradient contribution

Why NOT pre-train/fine-tune for tree models:
- XGBoost continue-training adds frozen trees, causing overfit on 187 samples
  (Eisenburger et al., 2024: GBDTs with noisy labels)
- Native sample_weight is the mechanistically correct approach

Training phases (all ablations for paper Table):
  Phase 1: Heuristic only (baseline) — already done in train.py
  Phase 2a: Heuristic + benchmark dev (weighted) — THIS SCRIPT
  Phase 2b: Benchmark-only 5-fold CV (ceiling per dimension)
  Phase 3: Cleanlab-denoised heuristic + benchmark dev (weighted)

Usage:
    python -m src.models.train_biquality
    python -m src.models.train_biquality --clean-weight 100 --model xgboost
    python -m src.models.train_biquality --model all --n-seeds 5 --save
    python -m src.models.train_biquality --benchmark-only  # Phase 2b ceiling
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

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]


def load_config():
    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        return yaml.safe_load(f)


def load_production_data(config):
    """Load production features + heuristic labels + temporal split."""
    paths = config["paths"]

    features = pd.read_parquet(PROJECT_DIR / paths["production_features"])
    labels = pd.read_parquet(PROJECT_DIR / paths["production_labels"])

    labels = labels.set_index("_jobid")
    features = features.set_index("_jobid")
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # Get feature columns (same exclusion logic as train.py)
    exclude = set(config.get("exclude_features", []))
    for col in features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in features.columns if c not in exclude]

    X = features[feature_cols].values.astype(np.float32)
    y = labels[DIMENSIONS].values.astype(np.float32)

    # Load temporal split
    split_path = PROJECT_DIR / paths["production_splits"]
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    train_idx = splits.get("train_idx", splits.get("train_indices"))
    val_idx = splits.get("val_idx", splits.get("val_indices"))

    return X, y, train_idx, val_idx, feature_cols


def load_benchmark_data(config, feature_cols):
    """Load benchmark features + GT labels + dev/test split."""
    paths = config["paths"]
    bench_dir = PROJECT_DIR / "data" / "processed" / "benchmark"

    features = pd.read_parquet(bench_dir / "features.parquet")
    labels = pd.read_parquet(bench_dir / "labels.parquet")

    # Align to same feature columns (fill missing with 0)
    X_cols = []
    for col in feature_cols:
        if col in features.columns:
            X_cols.append(features[col].values)
        else:
            X_cols.append(np.zeros(len(features)))
    X = np.column_stack(X_cols).astype(np.float32)
    y = labels[DIMENSIONS].values.astype(np.float32)

    # Load benchmark split (iterative stratification)
    split_path = bench_dir / "split_indices.pkl"
    if split_path.exists():
        with open(split_path, "rb") as f:
            bench_splits = pickle.load(f)
        dev_idx = bench_splits["dev_idx"]
        test_idx = bench_splits["test_idx"]
    else:
        logger.warning("No benchmark splits found. Run prepare_phase2_data.py first.")
        # Fallback: use all as test
        dev_idx = np.array([], dtype=int)
        test_idx = np.arange(len(X))

    return X, y, dev_idx, test_idx, labels


def compute_scale_pos_weight(y, max_weight=100.0):
    """Per-label scale_pos_weight."""
    weights = []
    for i in range(y.shape[1]):
        n_pos = y[:, i].sum()
        n_neg = len(y) - n_pos
        w = min(n_neg / max(n_pos, 1), max_weight)
        weights.append(w)
    return weights


def train_biquality(X_prod_train, y_prod_train, X_bench_dev, y_bench_dev,
                     X_val, y_val, model_type, config, clean_weight=100.0,
                     seed=42):
    """Train single-stage weighted combination model.

    Args:
        X_prod_train: Production training features (91K)
        y_prod_train: Production training labels (heuristic)
        X_bench_dev: Benchmark dev features (187)
        y_bench_dev: Benchmark dev labels (clean)
        X_val: Production validation features (for early stopping)
        y_val: Production validation labels
        model_type: 'xgboost', 'lightgbm', or 'random_forest'
        clean_weight: Weight multiplier for clean benchmark samples
        seed: Random seed
    """
    # Combine
    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])

    # Sample weights
    weights = np.ones(len(X_combined))
    weights[-len(X_bench_dev):] = clean_weight

    logger.info("  Combined: %d heuristic (w=1.0) + %d benchmark (w=%.0f) = %d total",
                len(X_prod_train), len(X_bench_dev), clean_weight, len(X_combined))

    # Per-label training
    spw = compute_scale_pos_weight(y_combined, max_weight=100.0)
    models = {}

    for i, dim in enumerate(DIMENSIONS):
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            params = config["models"]["xgboost"]["params"].copy()
            clf = XGBClassifier(
                **params, scale_pos_weight=spw[i],
                random_state=seed, verbosity=0,
            )
            clf.fit(
                X_combined, y_combined[:, i],
                sample_weight=weights,
                eval_set=[(X_val, y_val[:, i])],
                verbose=False,
            )
        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            params = config["models"]["lightgbm"]["params"].copy()
            clf = LGBMClassifier(
                **params, scale_pos_weight=spw[i],
                random_state=seed,
            )
            clf.fit(
                X_combined, y_combined[:, i],
                sample_weight=weights,
                eval_set=[(X_val, y_val[:, i])],
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            params = config["models"]["random_forest"]["params"].copy()
            clf = RandomForestClassifier(**params, random_state=seed)
            clf.fit(X_combined, y_combined[:, i], sample_weight=weights)
        elif model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            # MLP needs scaled features (unlike tree models)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            X_val_scaled = scaler.transform(X_val)
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
                verbose=False,
            )
            clf.fit(X_scaled, y_combined[:, i])
            # Store scaler with model for inference
            clf._ioprescriber_scaler = scaler
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        models[dim] = clf

    return models


def train_benchmark_only_cv(X_bench, y_bench, model_type, config, n_folds=5, seed=42):
    """Phase 2b: Benchmark-only cross-validation (ceiling analysis).

    Shows the best possible performance when training ONLY on clean data.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_bench)):
        X_tr, y_tr = X_bench[train_idx], y_bench[train_idx]
        X_vl, y_vl = X_bench[val_idx], y_bench[val_idx]

        # Train per-label
        y_pred = np.zeros_like(y_vl)
        for i, dim in enumerate(DIMENSIONS):
            if model_type == "xgboost":
                from xgboost import XGBClassifier
                params = config["models"]["xgboost"]["params"].copy()
                n_pos = y_tr[:, i].sum()
                spw = min((len(y_tr) - n_pos) / max(n_pos, 1), 100.0)
                clf = XGBClassifier(**params, scale_pos_weight=spw,
                                     random_state=seed, verbosity=0)
                clf.fit(X_tr, y_tr[:, i], verbose=False)
            elif model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                params = config["models"]["lightgbm"]["params"].copy()
                n_pos = y_tr[:, i].sum()
                spw = min((len(y_tr) - n_pos) / max(n_pos, 1), 100.0)
                clf = LGBMClassifier(**params, scale_pos_weight=spw,
                                      random_state=seed)
                clf.fit(X_tr, y_tr[:, i])
            else:
                from sklearn.ensemble import RandomForestClassifier
                params = config["models"]["random_forest"]["params"].copy()
                clf = RandomForestClassifier(**params, random_state=seed)
                clf.fit(X_tr, y_tr[:, i])
            y_pred[:, i] = clf.predict(X_vl)

        micro = f1_score(y_vl, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_vl, y_pred, average="macro", zero_division=0)
        fold_results.append({"fold": fold, "micro_f1": micro, "macro_f1": macro})
        logger.info("  Fold %d: Micro-F1=%.4f Macro-F1=%.4f", fold, micro, macro)

    mean_micro = np.mean([r["micro_f1"] for r in fold_results])
    mean_macro = np.mean([r["macro_f1"] for r in fold_results])
    std_micro = np.std([r["micro_f1"] for r in fold_results])
    std_macro = np.std([r["macro_f1"] for r in fold_results])

    return {
        "micro_f1_mean": mean_micro, "micro_f1_std": std_micro,
        "macro_f1_mean": mean_macro, "macro_f1_std": std_macro,
        "folds": fold_results,
    }


def evaluate_on_gt_test(models, X_test, y_test):
    """Evaluate on held-out benchmark test set."""
    from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

    y_pred = np.zeros_like(y_test)
    y_prob = np.zeros_like(y_test)
    for i, dim in enumerate(DIMENSIONS):
        # MLP models have a scaler attached — apply it before predict
        X_input = X_test
        if hasattr(models[dim], "_ioprescriber_scaler"):
            X_input = models[dim]._ioprescriber_scaler.transform(X_test)
        y_pred[:, i] = models[dim].predict(X_input)
        if hasattr(models[dim], "predict_proba"):
            y_prob[:, i] = models[dim].predict_proba(X_input)[:, 1]

    metrics = {
        "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_test, y_pred),
    }

    # Per-label
    metrics["per_label"] = {}
    for i, dim in enumerate(DIMENSIONS):
        n_pos = int(y_test[:, i].sum())
        metrics["per_label"][dim] = {
            "f1": f1_score(y_test[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_test[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_test[:, i], y_pred[:, i], zero_division=0),
            "support": n_pos,
        }

    # Bootstrap CI
    rng = np.random.RandomState(42)
    n = len(y_test)
    micro_boots, macro_boots = [], []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        micro_boots.append(f1_score(y_test[idx], y_pred[idx], average="micro", zero_division=0))
        macro_boots.append(f1_score(y_test[idx], y_pred[idx], average="macro", zero_division=0))
    metrics["micro_f1_ci"] = (np.percentile(micro_boots, 2.5), np.percentile(micro_boots, 97.5))
    metrics["macro_f1_ci"] = (np.percentile(macro_boots, 2.5), np.percentile(macro_boots, 97.5))

    return metrics, y_pred, y_prob


def log_results(name, metrics):
    """Log formatted results."""
    logger.info("")
    logger.info("=" * 75)
    logger.info("Results: %s", name)
    logger.info("=" * 75)
    logger.info("Micro-F1: %.4f [%.4f, %.4f]  Macro-F1: %.4f [%.4f, %.4f]  Hamming: %.4f",
                metrics["micro_f1"], metrics["micro_f1_ci"][0], metrics["micro_f1_ci"][1],
                metrics["macro_f1"], metrics["macro_f1_ci"][0], metrics["macro_f1_ci"][1],
                metrics["hamming_loss"])
    logger.info("")
    header = f"{'Dimension':<28s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Supp':>6s}"
    logger.info(header)
    logger.info("-" * len(header))
    for dim in DIMENSIONS:
        m = metrics["per_label"][dim]
        logger.info(f"{dim:<28s} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['support']:6d}")
    logger.info("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Biquality Learning Training")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm", "random_forest", "mlp", "all"])
    parser.add_argument("--clean-weight", type=float, default=100.0,
                        help="Weight multiplier for clean benchmark samples (default 100)")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run Phase 2b: benchmark-only CV (ceiling analysis)")
    parser.add_argument("--weight-search", action="store_true",
                        help="Search over clean weights [10, 50, 100, 200, 500]")
    args = parser.parse_args()

    config = load_config()
    SEEDS = [42, 123, 456, 789, 1024]
    seeds = SEEDS[:args.n_seeds]

    # Load data
    logger.info("Loading production data...")
    X_prod, y_prod, train_idx, val_idx, feature_cols = load_production_data(config)
    X_prod_train, y_prod_train = X_prod[train_idx], y_prod[train_idx]
    X_prod_val, y_prod_val = X_prod[val_idx], y_prod[val_idx]
    logger.info("Production: %d train, %d val, %d features",
                len(X_prod_train), len(X_prod_val), len(feature_cols))

    logger.info("Loading benchmark data...")
    X_bench, y_bench, dev_idx, test_idx, bench_labels = load_benchmark_data(config, feature_cols)
    X_bench_dev, y_bench_dev = X_bench[dev_idx], y_bench[dev_idx]
    X_bench_test, y_bench_test = X_bench[test_idx], y_bench[test_idx]
    logger.info("Benchmark: %d dev, %d test", len(dev_idx), len(test_idx))

    # Log GT label distribution
    logger.info("")
    logger.info("Benchmark test label distribution:")
    for i, dim in enumerate(DIMENSIONS):
        n = int(y_bench_test[:, i].sum())
        logger.info("  %-28s %4d (%5.1f%%)", dim, n, 100 * n / len(y_bench_test))

    model_types = ["xgboost", "lightgbm", "random_forest"] if args.model == "all" else [args.model]

    # ==============================
    # Phase 2b: Benchmark-only CV
    # ==============================
    if args.benchmark_only:
        logger.info("")
        logger.info("########################################")
        logger.info("PHASE 2b: Benchmark-Only CV (Ceiling)")
        logger.info("########################################")
        for mt in model_types:
            if not config["models"].get(mt, {}).get("enabled", False):
                continue
            logger.info("")
            logger.info("--- %s benchmark-only 5-fold CV ---", mt)
            cv_results = train_benchmark_only_cv(X_bench, y_bench, mt, config)
            logger.info("  Result: Micro-F1=%.4f+/-%.4f  Macro-F1=%.4f+/-%.4f",
                        cv_results["micro_f1_mean"], cv_results["micro_f1_std"],
                        cv_results["macro_f1_mean"], cv_results["macro_f1_std"])
        return

    # ==============================
    # Phase 2a: Biquality Training
    # ==============================
    if args.weight_search:
        weight_values = [10, 50, 100, 200, 500]
    else:
        weight_values = [args.clean_weight]

    all_results = {}

    for mt in model_types:
        if not config["models"].get(mt, {}).get("enabled", False):
            logger.info("Skipping %s (disabled)", mt)
            continue

        all_results[mt] = {}

        for w in weight_values:
            seed_metrics = []

            for seed in seeds:
                logger.info("")
                logger.info("Training %s (weight=%.0f, seed=%d)...", mt, w, seed)
                t0 = time.time()

                models = train_biquality(
                    X_prod_train, y_prod_train,
                    X_bench_dev, y_bench_dev,
                    X_prod_val, y_prod_val,
                    mt, config, clean_weight=w, seed=seed,
                )

                train_time = time.time() - t0
                logger.info("  Training completed in %.1fs", train_time)

                # Evaluate on benchmark test
                metrics, y_pred, y_prob = evaluate_on_gt_test(models, X_bench_test, y_bench_test)
                metrics["train_time"] = train_time
                metrics["seed"] = seed
                metrics["clean_weight"] = w

                if args.n_seeds == 1:
                    log_results(f"{mt} (w={w:.0f})", metrics)

                seed_metrics.append(metrics)

                # Save best model
                if args.save and seed == seeds[0]:
                    save_dir = PROJECT_DIR / "models" / "phase2"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{mt}_biquality_w{int(w)}.pkl"
                    with open(save_path, "wb") as f:
                        pickle.dump(models, f)
                    logger.info("  Saved to %s", save_path)

            # Aggregate seeds
            if len(seeds) > 1:
                mi = [m["micro_f1"] for m in seed_metrics]
                ma = [m["macro_f1"] for m in seed_metrics]
                logger.info("")
                logger.info("%s (w=%.0f, %d seeds): Micro-F1=%.4f+/-%.4f  Macro-F1=%.4f+/-%.4f",
                            mt, w, len(seeds), np.mean(mi), np.std(mi), np.mean(ma), np.std(ma))

            all_results[mt][w] = seed_metrics

    # Save all results
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "phase2_biquality_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    logger.info("")
    logger.info("All results saved to %s", results_path)

    # ==============================
    # Summary comparison table
    # ==============================
    if len(weight_values) > 1:
        logger.info("")
        logger.info("=" * 60)
        logger.info("WEIGHT SEARCH RESULTS")
        logger.info("=" * 60)
        for mt in model_types:
            if mt not in all_results:
                continue
            logger.info("")
            logger.info("--- %s ---", mt)
            header = f"  {'Weight':>8s} {'Micro-F1':>10s} {'Macro-F1':>10s}"
            logger.info(header)
            for w in weight_values:
                if w in all_results[mt]:
                    mi = np.mean([m["micro_f1"] for m in all_results[mt][w]])
                    ma = np.mean([m["macro_f1"] for m in all_results[mt][w]])
                    logger.info(f"  {w:8.0f} {mi:10.4f} {ma:10.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
