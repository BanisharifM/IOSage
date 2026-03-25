"""
Biquality weight sensitivity analysis for SC 2026 reviewer response.

Addresses W12: "Biquality weight=100 is used without justification."

Tests w in {1, 10, 50, 100, 200, 500} and reports Micro-F1 / Macro-F1
on the 436 GT test set. Generates a figure for the paper.

Usage:
    python scripts/run_weight_sensitivity.py
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

WEIGHTS_TO_TEST = [1, 10, 50, 100, 200, 500]
SEED = 42

# ---------- Style ----------
RCPARAMS_SC2026 = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'mathtext.fontset': 'stix',
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.fontsize': 7,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

COLORS = {
    'blue':      '#0072B2',
    'orange':    '#E69F00',
    'vermilion': '#D55E00',
}


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

    exclude = set(config.get("exclude_features", []))
    for col in features.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in features.columns if c not in exclude]

    X = features[feature_cols].values.astype(np.float32)
    y = labels[DIMENSIONS].values.astype(np.float32)

    split_path = PROJECT_DIR / paths["production_splits"]
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    train_idx = splits.get("train_idx", splits.get("train_indices"))
    val_idx = splits.get("val_idx", splits.get("val_indices"))

    return X, y, train_idx, val_idx, feature_cols


def load_benchmark_data(config, feature_cols):
    """Load benchmark features + GT labels + dev/test split."""
    bench_dir = PROJECT_DIR / "data" / "processed" / "benchmark"
    features = pd.read_parquet(bench_dir / "features.parquet")
    labels = pd.read_parquet(bench_dir / "labels.parquet")

    X_cols = []
    for col in feature_cols:
        if col in features.columns:
            X_cols.append(features[col].values)
        else:
            X_cols.append(np.zeros(len(features)))
    X = np.column_stack(X_cols).astype(np.float32)
    y = labels[DIMENSIONS].values.astype(np.float32)

    split_path = bench_dir / "split_indices.pkl"
    with open(split_path, "rb") as f:
        bench_splits = pickle.load(f)
    dev_idx = bench_splits["dev_idx"]
    test_idx = bench_splits["test_idx"]

    return X, y, dev_idx, test_idx


def compute_scale_pos_weight(y, max_weight=100.0):
    weights = []
    for i in range(y.shape[1]):
        n_pos = y[:, i].sum()
        n_neg = len(y) - n_pos
        w = min(n_neg / max(n_pos, 1), max_weight)
        weights.append(w)
    return weights


def train_and_evaluate(X_prod_train, y_prod_train, X_bench_dev, y_bench_dev,
                       X_val, y_val, X_test, y_test, config, clean_weight, seed):
    """Train XGBoost biquality model and evaluate on GT test set."""
    from xgboost import XGBClassifier

    X_combined = np.vstack([X_prod_train, X_bench_dev])
    y_combined = np.vstack([y_prod_train, y_bench_dev])

    sample_weights = np.ones(len(X_combined))
    sample_weights[-len(X_bench_dev):] = clean_weight

    spw = compute_scale_pos_weight(y_combined, max_weight=100.0)
    params = config["models"]["xgboost"]["params"].copy()

    y_pred = np.zeros_like(y_test)
    for i, dim in enumerate(DIMENSIONS):
        clf = XGBClassifier(
            **params, scale_pos_weight=spw[i],
            random_state=seed, verbosity=0,
        )
        clf.fit(
            X_combined, y_combined[:, i],
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        y_pred[:, i] = clf.predict(X_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-label F1
    per_label = {}
    for i, dim in enumerate(DIMENSIONS):
        per_label[dim] = float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0))

    return {
        "weight": clean_weight,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_label_f1": per_label,
    }


def make_figure(results, output_path):
    """Generate weight sensitivity figure."""
    plt.rcParams.update(RCPARAMS_SC2026)

    weights = [r["weight"] for r in results]
    micro = [r["micro_f1"] for r in results]
    macro = [r["macro_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    ax.plot(weights, micro, marker='o', color=COLORS['blue'],
            label='Micro-F1', linestyle='-', zorder=3)
    ax.plot(weights, macro, marker='s', color=COLORS['vermilion'],
            label='Macro-F1', linestyle='--', zorder=3)

    # Highlight w=100
    idx_100 = weights.index(100)
    ax.axvline(x=100, color=COLORS['orange'], linestyle=':', linewidth=0.8,
               alpha=0.7, zorder=1)
    ax.annotate(f'w=100\n({micro[idx_100]:.3f})',
                xy=(100, micro[idx_100]),
                xytext=(200, micro[idx_100] - 0.02),
                fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6),
                ha='left', va='top')

    ax.set_xscale('log')
    ax.set_xlabel('Benchmark sample weight (w)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(weights)
    ax.set_xticklabels([str(w) for w in weights])
    ax.legend(loc='lower right')

    # Set y limits with some padding
    all_vals = micro + macro
    ymin = min(all_vals) - 0.02
    ymax = max(all_vals) + 0.02
    ax.set_ylim(ymin, ymax)

    fig.savefig(output_path, format='pdf')
    fig.savefig(output_path.with_suffix('.png'), format='png', dpi=300)
    plt.close(fig)
    logger.info("Figure saved to %s", output_path)


def main():
    config = load_config()

    logger.info("Loading production data...")
    X_prod, y_prod, train_idx, val_idx, feature_cols = load_production_data(config)
    X_prod_train, y_prod_train = X_prod[train_idx], y_prod[train_idx]
    X_prod_val, y_prod_val = X_prod[val_idx], y_prod[val_idx]
    logger.info("Production: %d train, %d val, %d features",
                len(X_prod_train), len(X_prod_val), len(feature_cols))

    logger.info("Loading benchmark data...")
    X_bench, y_bench, dev_idx, test_idx = load_benchmark_data(config, feature_cols)
    X_bench_dev, y_bench_dev = X_bench[dev_idx], y_bench[dev_idx]
    X_bench_test, y_bench_test = X_bench[test_idx], y_bench[test_idx]
    logger.info("Benchmark: %d dev, %d test", len(dev_idx), len(test_idx))

    results = []
    for w in WEIGHTS_TO_TEST:
        logger.info("=" * 60)
        logger.info("Training with clean_weight = %d ...", w)
        t0 = time.time()
        result = train_and_evaluate(
            X_prod_train, y_prod_train,
            X_bench_dev, y_bench_dev,
            X_prod_val, y_prod_val,
            X_bench_test, y_bench_test,
            config, clean_weight=w, seed=SEED,
        )
        elapsed = time.time() - t0
        result["train_time_s"] = round(elapsed, 1)
        results.append(result)
        logger.info("  w=%d  Micro-F1=%.4f  Macro-F1=%.4f  (%.1fs)",
                     w, result["micro_f1"], result["macro_f1"], elapsed)

    # Summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("WEIGHT SENSITIVITY RESULTS (seed=%d)", SEED)
    logger.info("=" * 60)
    logger.info("  %8s  %10s  %10s  %8s", "Weight", "Micro-F1", "Macro-F1", "Time(s)")
    logger.info("  %8s  %10s  %10s  %8s", "------", "--------", "--------", "-------")
    best_micro_w = max(results, key=lambda r: r["micro_f1"])["weight"]
    best_macro_w = max(results, key=lambda r: r["macro_f1"])["weight"]
    for r in results:
        mi_mark = " *" if r["weight"] == best_micro_w else ""
        ma_mark = " *" if r["weight"] == best_macro_w else ""
        logger.info("  %8d  %10.4f%s  %10.4f%s  %8.1f",
                     r["weight"], r["micro_f1"], mi_mark, r["macro_f1"], ma_mark,
                     r["train_time_s"])
    logger.info("  (* = best)")
    logger.info("=" * 60)

    # Per-label breakdown
    logger.info("")
    logger.info("Per-label F1 by weight:")
    header = f"  {'Dimension':<28s}" + "".join(f"  w={w:>4d}" for w in WEIGHTS_TO_TEST)
    logger.info(header)
    logger.info("  " + "-" * (28 + 8 * len(WEIGHTS_TO_TEST)))
    for dim in DIMENSIONS:
        vals = "".join(f"  {r['per_label_f1'][dim]:6.3f}" for r in results)
        logger.info(f"  {dim:<28s}{vals}")

    # Save JSON
    results_path = PROJECT_DIR / "results" / "weight_sensitivity.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"seed": SEED, "weights": WEIGHTS_TO_TEST, "results": results}, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Generate figure
    fig_path = PROJECT_DIR / "paper" / "figures" / "fig_weight_sensitivity.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(results, fig_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
