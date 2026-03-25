#!/usr/bin/env python3
"""Generate paper figures and LaTeX tables for Track C iterative optimization results.

Figures (saved to paper/figures/iterative/):
  1. fig_convergence.pdf          -- Speedup vs iteration, per workload, faceted by model
  2. fig_single_vs_iterative.pdf  -- Track B vs Track C grouped bar chart (log-scale)
  3. fig_iterative_ablation.pdf   -- Ablation conditions bar chart
  4. fig_cost_vs_speedup.pdf      -- Cost-effectiveness scatter plot
  5. fig_model_comparison_iterative.pdf -- Model comparison (speedup + iters + cost)

Tables (saved to paper/tables/iterative/):
  1. tab_iterative_results.tex    -- Per-workload detailed results
  2. tab_iterative_models.tex     -- Per-model summary
  3. tab_iterative_ablation.tex   -- Ablation summary
  4. tab_trackb_vs_trackc.tex     -- Track B vs Track C comparison

Usage:
    python scripts/generate_iterative_figures.py
    python scripts/generate_iterative_figures.py --results-dir results/iterative
    python scripts/generate_iterative_figures.py --figures 1 3 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "paper" / "figures" / "iterative"
TAB_DIR = PROJECT_DIR / "paper" / "tables" / "iterative"

# ---------------------------------------------------------------------------
# Style Configuration (IEEE paper, consistent with generate_paper_figures.py)
# ---------------------------------------------------------------------------
STYLE_CONFIG = {
    "single_col": (3.5, 2.8),
    "double_col": (7.0, 3.0),
    "double_col_tall": (7.0, 4.5),
    "font_size": 8,
    "title_size": 9,
    "label_size": 8,
    "tick_size": 7,
    "legend_size": 7,
    "annotation_size": 6.5,
    # Colorblind-friendly palette (Okabe-Ito)
    "palette": {
        "primary": "#0072B2",    # blue
        "secondary": "#E69F00",  # orange
        "tertiary": "#D55E00",   # vermilion
        "green": "#009E73",
        "pink": "#CC79A7",
        "gray": "#BBBBBB",
        "cyan": "#56B4E9",
        "yellow": "#F0E442",
    },
    "grid_alpha": 0.3,
    "grid_style": "--",
    "spine_width": 0.5,
    "dpi": 300,
}

# Model display names and colors
MODEL_DISPLAY = {
    "claude-sonnet": "Claude Sonnet",
    "gpt-4o": "GPT-4o",
    "llama-70b": "Llama 3.1 70B",
}
MODEL_COLORS = {
    "claude-sonnet": "#0072B2",
    "gpt-4o": "#E69F00",
    "llama-70b": "#D55E00",
}

# Workload short names for figures
WORKLOAD_SHORT = {
    "ior_small_posix": "Small I/O",
    "ior_fsync_heavy": "Fsync",
    "ior_random_access": "Random",
    "ior_interface_shared": "Interface",
    "ior_misaligned": "Misaligned",
    "ior_small_direct": "O_DIRECT",
    "mdtest_metadata_storm": "Metadata",
    "ior_healthy_baseline": "Healthy",
}

# Ablation display names
ABLATION_DISPLAY = {
    "full": "Full System",
    "no_ml": "No ML",
    "no_kb": "No KB",
    "no_shap": "No SHAP",
    "single_shot": "Single-Shot",
    "no_feedback": "No Feedback",
}
ABLATION_ORDER = ["full", "no_ml", "no_kb", "no_shap", "single_shot", "no_feedback"]

# Track B workload name mapping to closed_loop_results.json keys
TRACKB_MAP = {
    "ior_small_posix": "ior_small_posix",
    "ior_fsync_heavy": "ior_fsync_heavy",
    "ior_random_access": "ior_random_to_sequential",
    "ior_interface_shared": "ior_interface_posix_shared",
}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
def apply_style():
    """Apply IEEE publication style to matplotlib."""
    cfg = STYLE_CONFIG
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": cfg["font_size"],
        "axes.titlesize": cfg["title_size"],
        "axes.labelsize": cfg["label_size"],
        "xtick.labelsize": cfg["tick_size"],
        "ytick.labelsize": cfg["tick_size"],
        "legend.fontsize": cfg["legend_size"],
        "figure.dpi": cfg["dpi"],
        "savefig.dpi": cfg["dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": cfg["grid_alpha"],
        "grid.linestyle": cfg["grid_style"],
        "axes.linewidth": cfg["spine_width"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, name):
    """Save figure as both PDF and PNG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    logger.info("  Saved: %s", pdf_path)


def save_table(content, name):
    """Save LaTeX table to file."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / name
    path.write_text(content)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_iterative_results(results_dir):
    """Load all iterative result JSON files from results_dir.

    Returns a list of individual run records (flattened from arrays).
    Each record has keys: workload, run_id, model, iterations, best_speedup, etc.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        return []

    records = []
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", results_dir)
        return []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", jf.name, exc)
            continue

        # Handle both single-object and array formats
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)

    logger.info("Loaded %d run records from %d files in %s",
                len(records), len(json_files), results_dir)
    return records


def load_trackb_results(path):
    """Load Track B closed-loop results."""
    path = Path(path)
    if not path.exists():
        logger.warning("Track B results not found: %s", path)
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load Track B results: %s", exc)
        return {}
    return data


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------
def compute_summary(records):
    """Compute summary metrics grouped by (workload, model, ablation).

    Returns a list of dicts, each with:
      workload, model, ablation, best_speedup, mean_speedup, std_speedup,
      mean_iterations, parse_success_rate, mean_cost, mean_tokens,
      mean_exec_time, final_statuses, n_runs, speedups (list),
      convergence_curves (list of lists)
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in records:
        workload = r.get("workload", "unknown")
        model = r.get("model", "unknown")
        # Determine ablation from config flags
        ablation = _infer_ablation(r)
        key = (workload, model, ablation)
        groups[key].append(r)

    summaries = []
    for (workload, model, ablation), runs in groups.items():
        speedups = []
        iterations_list = []
        parse_successes = []
        costs = []
        tokens = []
        exec_times = []
        statuses = []
        curves = []

        for run in runs:
            sp = run.get("best_speedup", 1.0)
            if sp is None or sp <= 0:
                sp = 1.0
            speedups.append(sp)

            iters = run.get("iterations", [])
            n_iters = len(iters)
            iterations_list.append(n_iters)

            # Parse success rate: fraction of iterations with valid LLM output
            if n_iters > 0:
                parse_ok = sum(
                    1 for it in iters
                    if it.get("parse_success", True)
                    and it.get("llm_response") is not None
                )
                parse_successes.append(parse_ok / n_iters)
            else:
                parse_successes.append(0.0)

            costs.append(run.get("total_cost_usd", 0.0) or 0.0)
            tokens.append(run.get("total_tokens", 0) or 0)
            exec_times.append(run.get("total_execution_time_s", 0.0) or 0.0)
            statuses.append(run.get("final_status", "unknown"))

            # Build convergence curve: speedup at each iteration
            curve = []
            for it in iters:
                it_sp = it.get("speedup", 1.0)
                if it_sp is None or it_sp <= 0:
                    it_sp = 1.0
                curve.append(it_sp)
            curves.append(curve)

        sp_arr = np.array(speedups)
        summaries.append({
            "workload": workload,
            "model": model,
            "ablation": ablation,
            "best_speedup": float(np.max(sp_arr)),
            "mean_speedup": float(np.mean(sp_arr)),
            "std_speedup": float(np.std(sp_arr)) if len(sp_arr) > 1 else 0.0,
            "geomean_speedup": float(np.exp(np.mean(np.log(sp_arr)))),
            "mean_iterations": float(np.mean(iterations_list)),
            "parse_success_rate": float(np.mean(parse_successes)),
            "mean_cost": float(np.mean(costs)),
            "mean_tokens": float(np.mean(tokens)),
            "mean_exec_time": float(np.mean(exec_times)),
            "final_statuses": statuses,
            "n_runs": len(runs),
            "speedups": speedups,
            "convergence_curves": curves,
        })

    return summaries


def _infer_ablation(record):
    """Infer the ablation condition from a run record's config flags."""
    cfg = record.get("config", {})
    if not cfg:
        return "full"

    use_ml = cfg.get("use_ml", True)
    use_shap = cfg.get("use_shap", True)
    use_kb = cfg.get("use_kb", True)
    use_feedback = cfg.get("use_feedback", True)
    max_iter = record.get("max_iterations", 5)

    if not use_ml and not use_shap:
        return "no_ml"
    if not use_kb:
        return "no_kb"
    if not use_shap and use_ml:
        return "no_shap"
    if not use_feedback:
        return "no_feedback"
    if max_iter == 1:
        return "single_shot"
    return "full"


def geomean(values):
    """Geometric mean of positive values. Returns 1.0 for empty input."""
    vals = [v for v in values if v > 0]
    if not vals:
        return 1.0
    return float(np.exp(np.mean(np.log(vals))))


def harmean(values):
    """Harmonic mean of positive values. Returns 0.0 for empty input."""
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return float(len(vals) / np.sum(1.0 / np.array(vals)))


# ---------------------------------------------------------------------------
# Figure 1: Convergence Curves
# ---------------------------------------------------------------------------
def fig_convergence(summaries):
    """Line plot: x=iteration, y=speedup, one line per workload, faceted by model."""
    logger.info("Generating fig_convergence...")

    # Group by model
    models_present = sorted(set(s["model"] for s in summaries if s["ablation"] == "full"))
    if not models_present:
        logger.warning("No data for convergence figure. Skipping.")
        return

    n_models = len(models_present)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 2.8),
                             squeeze=False, sharey=True)

    if HAS_SEABORN:
        colors = sns.color_palette("colorblind", n_colors=8)
    else:
        colors = [STYLE_CONFIG["palette"][k] for k in
                  ["primary", "secondary", "tertiary", "green", "pink", "cyan", "gray", "yellow"]]

    for midx, model in enumerate(models_present):
        ax = axes[0, midx]
        model_sums = [s for s in summaries
                      if s["model"] == model and s["ablation"] == "full"]

        for widx, s in enumerate(sorted(model_sums, key=lambda x: x["workload"])):
            wname = WORKLOAD_SHORT.get(s["workload"], s["workload"])
            curves = s["convergence_curves"]
            if not curves or all(len(c) == 0 for c in curves):
                continue

            # Average curve across runs (pad shorter curves with last value)
            max_len = max(len(c) for c in curves)
            padded = []
            for c in curves:
                if len(c) == 0:
                    continue
                padded_c = c + [c[-1]] * (max_len - len(c))
                padded.append(padded_c)

            if not padded:
                continue

            arr = np.array(padded)
            mean_curve = np.mean(arr, axis=0)
            iters = np.arange(1, max_len + 1)

            color = colors[widx % len(colors)]
            ax.plot(iters, mean_curve, marker="o", markersize=3,
                    linewidth=1.2, label=wname, color=color)

            if arr.shape[0] > 1:
                std_curve = np.std(arr, axis=0)
                ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                                alpha=0.15, color=color)

        ax.set_xlabel("Iteration")
        if midx == 0:
            ax.set_ylabel("Speedup")
        ax.set_title(MODEL_DISPLAY.get(model, model))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_xlim(left=0.8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(handles), 4), bbox_to_anchor=(0.5, -0.02),
                   fontsize=STYLE_CONFIG["legend_size"])

    fig.tight_layout()
    save_figure(fig, "fig_convergence")


# ---------------------------------------------------------------------------
# Figure 2: Track B vs Track C Comparison
# ---------------------------------------------------------------------------
def fig_single_vs_iterative(summaries, trackb_data):
    """Grouped bar chart: Track B (single-shot) vs Track C (iterative) speedup."""
    logger.info("Generating fig_single_vs_iterative...")

    # Get full-ablation results for the best model (or all if only one)
    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums:
        logger.warning("No full-ablation data. Skipping single_vs_iterative.")
        return

    # Pick the best model by geomean speedup
    model_geomeans = {}
    for s in full_sums:
        model_geomeans.setdefault(s["model"], []).append(s["geomean_speedup"])
    best_model = max(model_geomeans, key=lambda m: geomean(model_geomeans[m]))

    best_sums = {s["workload"]: s for s in full_sums if s["model"] == best_model}

    # Collect workloads present in both Track B and Track C
    workloads = []
    trackb_speeds = []
    trackc_speeds = []
    trackc_stds = []

    for wl, tb_key in TRACKB_MAP.items():
        tb_entry = trackb_data.get(tb_key, {})
        tc_entry = best_sums.get(wl)

        tb_sp = tb_entry.get("speedup_write", None)
        if tb_sp is None or tb_sp <= 0:
            continue
        if tb_entry.get("status") == "complete_but_excluded":
            continue

        tc_sp = tc_entry["mean_speedup"] if tc_entry else None
        tc_std = tc_entry["std_speedup"] if tc_entry else 0.0

        workloads.append(WORKLOAD_SHORT.get(wl, wl))
        trackb_speeds.append(tb_sp)
        trackc_speeds.append(tc_sp if tc_sp else 1.0)
        trackc_stds.append(tc_std)

    # Also add Track C-only workloads
    for wl, s in best_sums.items():
        if wl in TRACKB_MAP:
            continue
        workloads.append(WORKLOAD_SHORT.get(wl, wl))
        trackb_speeds.append(None)
        trackc_speeds.append(s["mean_speedup"])
        trackc_stds.append(s["std_speedup"])

    if not workloads:
        logger.warning("No overlapping workloads for Track B vs C. Skipping.")
        return

    x = np.arange(len(workloads))
    width = 0.35

    fig, ax = plt.subplots(figsize=STYLE_CONFIG["double_col"])
    cfg = STYLE_CONFIG

    # Track B bars
    tb_vals = [v if v is not None else 0 for v in trackb_speeds]
    tb_mask = [v is not None for v in trackb_speeds]
    bars_b = ax.bar(x - width / 2, tb_vals, width,
                    label="Track B (single-shot)",
                    color=cfg["palette"]["secondary"], edgecolor="white",
                    linewidth=0.5)

    # Track C bars
    bars_c = ax.bar(x + width / 2, trackc_speeds, width, yerr=trackc_stds,
                    label="Track C (iterative)",
                    color=cfg["palette"]["primary"], edgecolor="white",
                    linewidth=0.5, capsize=3)

    # Gray out missing Track B bars
    for i, present in enumerate(tb_mask):
        if not present:
            bars_b[i].set_color(cfg["palette"]["gray"])
            bars_b[i].set_alpha(0.3)

    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=cfg["legend_size"])
    ax.set_title("Track B (Single-Shot) vs Track C (Iterative) Speedup")

    fig.tight_layout()
    save_figure(fig, "fig_single_vs_iterative")


# ---------------------------------------------------------------------------
# Figure 3: Ablation Study
# ---------------------------------------------------------------------------
def fig_iterative_ablation(summaries):
    """Bar chart: ablation conditions vs geometric mean speedup."""
    logger.info("Generating fig_iterative_ablation...")

    ablation_data = {}
    for s in summaries:
        abl = s["ablation"]
        ablation_data.setdefault(abl, []).extend(s["speedups"])

    labels = []
    means = []
    errs_lo = []
    errs_hi = []

    for abl in ABLATION_ORDER:
        speeds = ablation_data.get(abl, [])
        if not speeds:
            continue
        gm = geomean(speeds)
        labels.append(ABLATION_DISPLAY.get(abl, abl))
        means.append(gm)
        # Bootstrap CI for geomean (simple percentile)
        if len(speeds) > 1:
            rng = np.random.default_rng(42)
            boot_gm = []
            for _ in range(1000):
                sample = rng.choice(speeds, size=len(speeds), replace=True)
                sample = np.maximum(sample, 1e-10)
                boot_gm.append(np.exp(np.mean(np.log(sample))))
            lo = np.percentile(boot_gm, 2.5)
            hi = np.percentile(boot_gm, 97.5)
            errs_lo.append(gm - lo)
            errs_hi.append(hi - gm)
        else:
            errs_lo.append(0.0)
            errs_hi.append(0.0)

    if not labels:
        logger.warning("No ablation data. Skipping.")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=STYLE_CONFIG["double_col"])

    if HAS_SEABORN:
        colors = sns.color_palette("colorblind", n_colors=len(labels))
    else:
        colors = [STYLE_CONFIG["palette"]["primary"]] * len(labels)

    ax.bar(x, means, yerr=[errs_lo, errs_hi], capsize=4,
           color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Geometric Mean Speedup")
    ax.set_title("Ablation Study: Component Contributions")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))

    # Annotate values on bars
    for i, (m, label) in enumerate(zip(means, labels)):
        ax.text(i, m + errs_hi[i] + 0.05 * max(means),
                f"{m:.2f}x", ha="center", va="bottom",
                fontsize=STYLE_CONFIG["annotation_size"])

    fig.tight_layout()
    save_figure(fig, "fig_iterative_ablation")


# ---------------------------------------------------------------------------
# Figure 4: Cost vs Speedup
# ---------------------------------------------------------------------------
def fig_cost_vs_speedup(summaries):
    """Scatter plot: cost (USD) vs speedup per workload per model."""
    logger.info("Generating fig_cost_vs_speedup...")

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums:
        logger.warning("No full-ablation data. Skipping cost_vs_speedup.")
        return

    fig, ax = plt.subplots(figsize=STYLE_CONFIG["single_col"])

    for s in full_sums:
        model = s["model"]
        color = MODEL_COLORS.get(model, STYLE_CONFIG["palette"]["primary"])
        label = MODEL_DISPLAY.get(model, model)
        wname = WORKLOAD_SHORT.get(s["workload"], s["workload"])

        ax.scatter(s["mean_cost"], s["geomean_speedup"],
                   color=color, s=40, alpha=0.8, edgecolors="white",
                   linewidth=0.3, label=label, zorder=3)
        ax.annotate(wname, (s["mean_cost"], s["geomean_speedup"]),
                    fontsize=STYLE_CONFIG["annotation_size"],
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Speedup")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_title("Cost-Effectiveness")

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = True
            unique_h.append(h)
            unique_l.append(lbl)
    ax.legend(unique_h, unique_l, fontsize=STYLE_CONFIG["legend_size"])

    fig.tight_layout()
    save_figure(fig, "fig_cost_vs_speedup")


# ---------------------------------------------------------------------------
# Figure 5: Model Comparison (Multi-metric)
# ---------------------------------------------------------------------------
def fig_model_comparison_iterative(summaries):
    """Grouped bar chart: models x (geomean speedup, mean iters, mean cost)."""
    logger.info("Generating fig_model_comparison_iterative...")

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums:
        logger.warning("No full-ablation data. Skipping model_comparison.")
        return

    # Aggregate per model
    from collections import defaultdict
    model_data = defaultdict(lambda: {"speedups": [], "iters": [], "costs": []})
    for s in full_sums:
        model_data[s["model"]]["speedups"].extend(s["speedups"])
        model_data[s["model"]]["iters"].append(s["mean_iterations"])
        model_data[s["model"]]["costs"].append(s["mean_cost"])

    models = sorted(model_data.keys())
    if not models:
        logger.warning("No model data. Skipping.")
        return

    gm_speedups = [geomean(model_data[m]["speedups"]) for m in models]
    mean_iters = [float(np.mean(model_data[m]["iters"])) for m in models]
    mean_costs = [float(np.mean(model_data[m]["costs"])) for m in models]
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    cfg = STYLE_CONFIG
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, cfg["palette"]["primary"]) for m in models]

    # Panel (a): Geomean speedup
    axes[0].bar(x, gm_speedups, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=30, ha="right")
    axes[0].set_ylabel("Geo. Mean Speedup")
    axes[0].set_title("(a) Speedup")
    axes[0].axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    for i, v in enumerate(gm_speedups):
        axes[0].text(i, v + 0.02 * max(gm_speedups), f"{v:.2f}x",
                     ha="center", va="bottom", fontsize=cfg["annotation_size"])

    # Panel (b): Mean iterations
    axes[1].bar(x, mean_iters, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Iterations")
    axes[1].set_title("(b) Iterations")
    for i, v in enumerate(mean_iters):
        axes[1].text(i, v + 0.05, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=cfg["annotation_size"])

    # Panel (c): Mean cost
    axes[2].bar(x, mean_costs, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_names, rotation=30, ha="right")
    axes[2].set_ylabel("Mean Cost (USD)")
    axes[2].set_title("(c) Cost")
    for i, v in enumerate(mean_costs):
        axes[2].text(i, v + 0.01 * max(mean_costs) if mean_costs else 0,
                     f"${v:.3f}", ha="center", va="bottom",
                     fontsize=cfg["annotation_size"])

    fig.tight_layout()
    save_figure(fig, "fig_model_comparison_iterative")


# ---------------------------------------------------------------------------
# Table 1: Per-Workload Results
# ---------------------------------------------------------------------------
def tab_iterative_results(summaries):
    """LaTeX table: per-workload results for the best model."""
    logger.info("Generating tab_iterative_results...")

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums:
        logger.warning("No full-ablation data. Skipping tab_iterative_results.")
        return

    # Pick best model by geomean
    from collections import defaultdict
    model_speeds = defaultdict(list)
    for s in full_sums:
        model_speeds[s["model"]].extend(s["speedups"])
    best_model = max(model_speeds, key=lambda m: geomean(model_speeds[m]))

    rows = [s for s in full_sums if s["model"] == best_model]
    rows.sort(key=lambda s: s["workload"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Track C iterative optimization results per workload")
    lines.append(r"(" + MODEL_DISPLAY.get(best_model, best_model) + r").}")
    lines.append(r"\label{tab:iterative_results}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{llrrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Workload} & \textbf{Bottleneck} & \textbf{Speedup} "
                 r"& \textbf{Iters} & \textbf{Cost} & \textbf{Status} \\")
    lines.append(r"\midrule")

    speedups_all = []
    for s in rows:
        wname = WORKLOAD_SHORT.get(s["workload"], s["workload"])
        sp_str = f'{s["mean_speedup"]:.1f}$\\times$'
        if s["std_speedup"] > 0:
            sp_str = f'{s["mean_speedup"]:.1f}$\\pm${s["std_speedup"]:.1f}$\\times$'
        iters_str = f'{s["mean_iterations"]:.1f}'
        cost_str = f'\\${s["mean_cost"]:.3f}'
        # Most common status
        from collections import Counter
        status_counts = Counter(s["final_statuses"])
        status = status_counts.most_common(1)[0][0]
        status_map = {
            "converged": "Conv.",
            "plateau": "Plateau",
            "max_iterations": "Max iter.",
            "baseline_failed": "BL fail",
            "unknown": "--",
        }
        status_str = status_map.get(status, status)
        bottleneck = s["workload"].replace("ior_", "").replace("mdtest_", "").replace("_", " ").title()

        lines.append(f"{wname} & {bottleneck} & {sp_str} & {iters_str} "
                     f"& {cost_str} & {status_str} \\\\")
        speedups_all.extend(s["speedups"])

    lines.append(r"\midrule")
    gm = geomean(speedups_all)
    hm = harmean(speedups_all)
    lines.append(f"\\multicolumn{{2}}{{l}}{{Geometric mean}} & "
                 f"\\textbf{{{gm:.1f}$\\times$}} & & & \\\\")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Harmonic mean}} & "
                 f"\\textbf{{{hm:.1f}$\\times$}} & & & \\\\")
    if speedups_all:
        lines.append(f"\\multicolumn{{2}}{{l}}{{Range}} & "
                     f"{min(speedups_all):.1f}--{max(speedups_all):.1f}$\\times$ & & & \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_iterative_results.tex")


# ---------------------------------------------------------------------------
# Table 2: Per-Model Summary
# ---------------------------------------------------------------------------
def tab_iterative_models(summaries):
    """LaTeX table: per-model summary statistics."""
    logger.info("Generating tab_iterative_models...")

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums:
        logger.warning("No full-ablation data. Skipping tab_iterative_models.")
        return

    from collections import defaultdict
    model_data = defaultdict(lambda: {
        "speedups": [], "iters": [], "costs": [], "parse": [], "statuses": []
    })
    for s in full_sums:
        model_data[s["model"]]["speedups"].extend(s["speedups"])
        model_data[s["model"]]["iters"].append(s["mean_iterations"])
        model_data[s["model"]]["costs"].append(s["mean_cost"])
        model_data[s["model"]]["parse"].append(s["parse_success_rate"])
        model_data[s["model"]]["statuses"].extend(s["final_statuses"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Track C iterative optimization: per-model summary.}")
    lines.append(r"\label{tab:iterative_models}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Geo. Sp.} & \textbf{Iters} "
                 r"& \textbf{Cost} & \textbf{Parse\%} & \textbf{Impr.\%} \\")
    lines.append(r"\midrule")

    for model in sorted(model_data.keys()):
        d = model_data[model]
        gm = geomean(d["speedups"])
        mean_it = float(np.mean(d["iters"]))
        mean_cost = float(np.mean(d["costs"]))
        parse_pct = float(np.mean(d["parse"])) * 100
        impr_rate = sum(1 for sp in d["speedups"] if sp > 1.0) / max(len(d["speedups"]), 1) * 100

        name = MODEL_DISPLAY.get(model, model)
        lines.append(f"{name} & {gm:.2f}$\\times$ & {mean_it:.1f} "
                     f"& \\${mean_cost:.3f} & {parse_pct:.0f}\\% & {impr_rate:.0f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_iterative_models.tex")


# ---------------------------------------------------------------------------
# Table 3: Ablation Summary
# ---------------------------------------------------------------------------
def tab_iterative_ablation(summaries):
    """LaTeX table: ablation condition summary."""
    logger.info("Generating tab_iterative_ablation...")

    from collections import defaultdict
    abl_data = defaultdict(lambda: {"speedups": [], "iters": [], "statuses": []})
    for s in summaries:
        abl_data[s["ablation"]]["speedups"].extend(s["speedups"])
        abl_data[s["ablation"]]["iters"].append(s["mean_iterations"])
        abl_data[s["ablation"]]["statuses"].extend(s["final_statuses"])

    if not abl_data:
        logger.warning("No ablation data. Skipping tab_iterative_ablation.")
        return

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Ablation study: contribution of each component.}")
    lines.append(r"\label{tab:iterative_ablation}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Condition} & \textbf{Geo. Speedup} "
                 r"& \textbf{Mean Iters} & \textbf{Impr.\%} \\")
    lines.append(r"\midrule")

    for abl in ABLATION_ORDER:
        if abl not in abl_data:
            continue
        d = abl_data[abl]
        gm = geomean(d["speedups"])
        mean_it = float(np.mean(d["iters"]))
        impr_rate = sum(1 for sp in d["speedups"] if sp > 1.0) / max(len(d["speedups"]), 1) * 100
        name = ABLATION_DISPLAY.get(abl, abl)
        lines.append(f"{name} & {gm:.2f}$\\times$ & {mean_it:.1f} "
                     f"& {impr_rate:.0f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_iterative_ablation.tex")


# ---------------------------------------------------------------------------
# Table 4: Track B vs Track C
# ---------------------------------------------------------------------------
def tab_trackb_vs_trackc(summaries, trackb_data):
    """LaTeX table comparing Track B and Track C results."""
    logger.info("Generating tab_trackb_vs_trackc...")

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    if not full_sums and not trackb_data:
        logger.warning("No data for Track B vs C. Skipping.")
        return

    # Best model from Track C
    from collections import defaultdict
    model_speeds = defaultdict(list)
    for s in full_sums:
        model_speeds[s["model"]].extend(s["speedups"])
    best_model = max(model_speeds, key=lambda m: geomean(model_speeds[m])) if model_speeds else None

    tc_by_wl = {}
    if best_model:
        tc_by_wl = {s["workload"]: s for s in full_sums if s["model"] == best_model}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Comparison: single-shot (Track B) vs iterative (Track C).}")
    lines.append(r"\label{tab:trackb_vs_trackc}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Workload} & \textbf{Track B} & \textbf{Track C} "
                 r"& \textbf{Impr.} \\")
    lines.append(r"\midrule")

    tb_speeds = []
    tc_speeds = []

    all_wl = set(TRACKB_MAP.keys()) | set(tc_by_wl.keys())
    for wl in sorted(all_wl):
        wname = WORKLOAD_SHORT.get(wl, wl)

        tb_key = TRACKB_MAP.get(wl)
        tb_entry = trackb_data.get(tb_key, {}) if tb_key else {}
        tb_sp = tb_entry.get("speedup_write")
        if tb_entry.get("status") == "complete_but_excluded":
            tb_sp = None

        tc_entry = tc_by_wl.get(wl)
        tc_sp = tc_entry["mean_speedup"] if tc_entry else None

        tb_str = f"{tb_sp:.1f}$\\times$" if tb_sp else "--"
        tc_str = f"{tc_sp:.1f}$\\times$" if tc_sp else "--"

        if tb_sp and tc_sp:
            impr = tc_sp / tb_sp
            impr_str = f"{impr:.2f}$\\times$"
            tb_speeds.append(tb_sp)
            tc_speeds.append(tc_sp)
        else:
            impr_str = "--"

        lines.append(f"{wname} & {tb_str} & {tc_str} & {impr_str} \\\\")

    lines.append(r"\midrule")
    if tb_speeds:
        tb_gm = geomean(tb_speeds)
        tc_gm = geomean(tc_speeds)
        lines.append(f"Geo. mean & {tb_gm:.1f}$\\times$ & {tc_gm:.1f}$\\times$ "
                     f"& {tc_gm / tb_gm:.2f}$\\times$ \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_table("\n".join(lines), "tab_trackb_vs_trackc.tex")


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------
def print_summary(summaries, trackb_data):
    """Print summary statistics to stdout."""
    print("\n" + "=" * 70)
    print("TRACK C ITERATIVE OPTIMIZATION -- SUMMARY STATISTICS")
    print("=" * 70)

    if not summaries:
        print("No iterative results loaded.")
        return

    full_sums = [s for s in summaries if s["ablation"] == "full"]
    all_speedups = []
    for s in full_sums:
        all_speedups.extend(s["speedups"])

    print(f"\nTotal run records: {sum(s['n_runs'] for s in summaries)}")
    print(f"Full-ablation records: {sum(s['n_runs'] for s in full_sums)}")
    print(f"Unique workloads: {len(set(s['workload'] for s in summaries))}")
    print(f"Unique models: {len(set(s['model'] for s in summaries))}")
    print(f"Unique ablations: {len(set(s['ablation'] for s in summaries))}")

    if all_speedups:
        print(f"\n--- Full System (all models, all workloads) ---")
        print(f"Geometric mean speedup: {geomean(all_speedups):.2f}x")
        print(f"Harmonic mean speedup:  {harmean(all_speedups):.2f}x")
        print(f"Arithmetic mean:        {np.mean(all_speedups):.2f}x (for reference only)")
        print(f"Range:                  {min(all_speedups):.1f}x -- {max(all_speedups):.1f}x")

    # Per-model breakdown
    from collections import defaultdict
    model_speeds = defaultdict(list)
    for s in full_sums:
        model_speeds[s["model"]].extend(s["speedups"])

    if model_speeds:
        print(f"\n--- Per-Model Geometric Mean ---")
        for model in sorted(model_speeds):
            gm = geomean(model_speeds[model])
            print(f"  {MODEL_DISPLAY.get(model, model):20s}: {gm:.2f}x "
                  f"(n={len(model_speeds[model])})")

    # Per-ablation breakdown
    abl_speeds = defaultdict(list)
    for s in summaries:
        abl_speeds[s["ablation"]].extend(s["speedups"])

    if len(abl_speeds) > 1:
        print(f"\n--- Per-Ablation Geometric Mean ---")
        for abl in ABLATION_ORDER:
            if abl in abl_speeds:
                gm = geomean(abl_speeds[abl])
                print(f"  {ABLATION_DISPLAY.get(abl, abl):20s}: {gm:.2f}x "
                      f"(n={len(abl_speeds[abl])})")

    # Track B comparison
    if trackb_data:
        tb_summary = trackb_data.get("summary", {})
        if tb_summary:
            print(f"\n--- Track B Reference ---")
            print(f"Track B geometric mean: {tb_summary.get('geometric_mean_write', 'N/A')}x")
            print(f"Track B harmonic mean:  {tb_summary.get('harmonic_mean_write', 'N/A')}x")
            print(f"Track B range:          {tb_summary.get('range', 'N/A')}")

    # Status distribution
    all_statuses = []
    for s in full_sums:
        all_statuses.extend(s["final_statuses"])
    if all_statuses:
        from collections import Counter
        print(f"\n--- Status Distribution (full ablation) ---")
        for status, count in Counter(all_statuses).most_common():
            print(f"  {status:20s}: {count}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Track C iterative optimization figures and tables."
    )
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_DIR / "results" / "iterative"),
                        help="Directory containing iterative result JSON files")
    parser.add_argument("--trackb-results", type=str,
                        default=str(PROJECT_DIR / "results" / "closed_loop" / "closed_loop_results.json"),
                        help="Path to Track B closed-loop results JSON")
    parser.add_argument("--figures", nargs="*", type=int, default=None,
                        help="Generate only specific figures (1-5). Default: all.")
    args = parser.parse_args()

    apply_style()

    # Load data
    records = load_iterative_results(args.results_dir)
    trackb_data = load_trackb_results(args.trackb_results)

    if not records:
        logger.warning("No iterative results found. Generating empty placeholder outputs.")
        # Still generate Track B tables if available
        if trackb_data:
            tab_trackb_vs_trackc([], trackb_data)
        print_summary([], trackb_data)
        return

    # Compute summaries
    summaries = compute_summary(records)
    logger.info("Computed summaries for %d (workload, model, ablation) groups.", len(summaries))

    # Generate figures
    fig_map = {
        1: lambda: fig_convergence(summaries),
        2: lambda: fig_single_vs_iterative(summaries, trackb_data),
        3: lambda: fig_iterative_ablation(summaries),
        4: lambda: fig_cost_vs_speedup(summaries),
        5: lambda: fig_model_comparison_iterative(summaries),
    }

    figs_to_gen = args.figures if args.figures else list(fig_map.keys())
    for fig_num in figs_to_gen:
        if fig_num in fig_map:
            try:
                fig_map[fig_num]()
            except Exception as exc:
                logger.error("Failed to generate figure %d: %s", fig_num, exc,
                             exc_info=True)
        else:
            logger.warning("Unknown figure number: %d (valid: 1-5)", fig_num)

    # Generate tables
    try:
        tab_iterative_results(summaries)
    except Exception as exc:
        logger.error("Failed tab_iterative_results: %s", exc, exc_info=True)

    try:
        tab_iterative_models(summaries)
    except Exception as exc:
        logger.error("Failed tab_iterative_models: %s", exc, exc_info=True)

    try:
        tab_iterative_ablation(summaries)
    except Exception as exc:
        logger.error("Failed tab_iterative_ablation: %s", exc, exc_info=True)

    try:
        tab_trackb_vs_trackc(summaries, trackb_data)
    except Exception as exc:
        logger.error("Failed tab_trackb_vs_trackc: %s", exc, exc_info=True)

    # Print summary
    print_summary(summaries, trackb_data)

    logger.info("All outputs saved to:")
    logger.info("  Figures: %s", FIG_DIR)
    logger.info("  Tables:  %s", TAB_DIR)


if __name__ == "__main__":
    main()
