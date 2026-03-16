#!/usr/bin/env python3
"""
Generate Paper-Ready Figures for SC 2026 Labeling Section
=========================================================
Produces publication-quality figures for the heuristic labeling results.

Output directory: paper/figures/labeling/

Figure inventory:
  L1. fig_heuristic_label_distribution.pdf  — 8-dimension bar chart with counts/rates
  L2. fig_multilabel_cooccurrence.pdf       — Co-occurrence heatmap (which bottlenecks co-occur)
  L3. fig_drishti_code_rates.pdf            — All 30 Drishti codes trigger rates (sorted)
  L4. fig_confidence_histogram.pdf          — Confidence score distribution

Usage:
    python scripts/generate_labeling_figures.py
    python scripts/generate_labeling_figures.py --figures L1 L3
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_config import (
    apply_style, save_figure, add_panel_label, format_count,
    SINGLE_COL, DOUBLE_COL, DOUBLE_COL_TALL,
    PALETTE_8, COLORS, HATCHES,
    DIMENSION_ORDER, DIMENSION_LABELS, DIMENSION_LABELS_SHORT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUT_DIR = PROJECT_DIR / 'paper' / 'figures' / 'labeling'

# Drishti insight codes grouped by severity
DRISHTI_CODES = {
    'HIGH': ['drishti_S01', 'drishti_P05', 'drishti_P06', 'drishti_P07',
             'drishti_P08', 'drishti_P11', 'drishti_P13', 'drishti_P15',
             'drishti_P16', 'drishti_P17', 'drishti_P18', 'drishti_P19',
             'drishti_P21', 'drishti_P22', 'drishti_M02', 'drishti_M03'],
    'WARN': ['drishti_M01', 'drishti_M06', 'drishti_M07',
             'drishti_P09', 'drishti_P10'],
    'INFO': ['drishti_P01', 'drishti_P02', 'drishti_P03', 'drishti_P04'],
    'OK':   ['drishti_P12', 'drishti_P14', 'drishti_M04', 'drishti_M05'],
}

# Human-readable code descriptions
CODE_DESCRIPTIONS = {
    'drishti_S01': 'STDIO high usage',
    'drishti_P05': 'Small reads',
    'drishti_P06': 'Small writes',
    'drishti_P07': 'Misaligned memory',
    'drishti_P08': 'Misaligned file',
    'drishti_P11': 'Random reads',
    'drishti_P13': 'Random writes',
    'drishti_P15': 'Small shared reads',
    'drishti_P16': 'Small shared writes',
    'drishti_P17': 'High metadata time',
    'drishti_P18': 'Shared data imbalance',
    'drishti_P19': 'Shared time imbalance',
    'drishti_P21': 'Write size imbalance',
    'drishti_P22': 'Read size imbalance',
    'drishti_M02': 'No collective reads',
    'drishti_M03': 'No collective writes',
    'drishti_M01': 'No MPI-IO usage',
    'drishti_M06': 'No non-blocking reads',
    'drishti_M07': 'No non-blocking writes',
    'drishti_P09': 'Redundant reads',
    'drishti_P10': 'Redundant writes',
    'drishti_P01': 'Write intensive (ops)',
    'drishti_P02': 'Read intensive (ops)',
    'drishti_P03': 'Write intensive (bytes)',
    'drishti_P04': 'Read intensive (bytes)',
    'drishti_P12': 'Sequential reads',
    'drishti_P14': 'Sequential writes',
    'drishti_M04': 'Collective reads',
    'drishti_M05': 'Collective writes',
}

SEVERITY_COLORS = {
    'HIGH': COLORS['vermilion'],
    'WARN': COLORS['orange'],
    'INFO': COLORS['cyan'],
    'OK':   COLORS['green'],
}


def load_heuristic_labels():
    """Load heuristic labels parquet."""
    path = DATA_DIR / 'heuristic_labels.parquet'
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    logger.info(f"Loaded heuristic labels: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ===================================================================
# L1: Heuristic Label Distribution (8 Dimensions)
# ===================================================================
def fig_L1_label_distribution(df):
    """Bar chart showing positive rate and count for each dimension."""
    logger.info("Generating L1: Heuristic label distribution...")

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    counts = []
    rates = []
    for dim in DIMENSION_ORDER:
        if dim in df.columns:
            c = int(df[dim].sum())
            r = c / len(df) * 100
        else:
            c, r = 0, 0.0
        counts.append(c)
        rates.append(r)

    x = np.arange(len(DIMENSION_ORDER))
    bars = ax.bar(x, rates, color=PALETTE_8, edgecolor='black',
                  linewidth=0.3, zorder=3)

    # Add hatching for B&W readability
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)

    # Annotate with count above each bar
    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(i, rate + 1.2, format_count(count),
                ha='center', va='bottom', fontsize=6.5, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS_SHORT[d] for d in DIMENSION_ORDER],
                       fontsize=6.5, rotation=35, ha='right')
    ax.set_ylabel('Positive Rate (%)')
    ax.set_ylim(0, max(rates) * 1.25)

    # Add a subtle horizontal line at key thresholds
    for threshold in [5, 20]:
        if threshold < max(rates):
            ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.4,
                       linewidth=0.5, zorder=1)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'fig_heuristic_label_distribution')
    return counts, rates


# ===================================================================
# L2: Multi-Label Co-occurrence Heatmap
# ===================================================================
def fig_L2_cooccurrence(df):
    """Heatmap showing how often bottleneck dimensions co-occur."""
    logger.info("Generating L2: Multi-label co-occurrence...")

    dims = [d for d in DIMENSION_ORDER if d in df.columns]
    n = len(dims)

    # Compute co-occurrence matrix (count of samples with both dims active)
    cooccur = np.zeros((n, n), dtype=int)
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            cooccur[i, j] = int(((df[d1] == 1) & (df[d2] == 1)).sum())

    # Normalize by minimum of the two marginals (Jaccard-like)
    # Use percentage of co-occurrence relative to the smaller class
    cooccur_pct = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            min_count = min(cooccur[i, i], cooccur[j, j])
            if min_count > 0 and i != j:
                cooccur_pct[i, j] = cooccur[i, j] / min_count * 100
            elif i == j:
                cooccur_pct[i, j] = 100.0

    # Mask upper triangle for cleaner look
    mask = np.triu(np.ones_like(cooccur_pct, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    sns.heatmap(
        cooccur_pct, mask=mask,
        xticklabels=[DIMENSION_LABELS_SHORT[d] for d in dims],
        yticklabels=[DIMENSION_LABELS_SHORT[d] for d in dims],
        annot=True, fmt='.0f', annot_kws={'size': 6},
        cmap='YlOrRd', vmin=0, vmax=100,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Co-occurrence (%)', 'shrink': 0.8},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'fig_multilabel_cooccurrence')


# ===================================================================
# L3: Drishti Code Trigger Rates
# ===================================================================
def fig_L3_drishti_codes(df):
    """Horizontal bar chart of all 30 Drishti insight code trigger rates."""
    logger.info("Generating L3: Drishti code trigger rates...")

    # Collect trigger rates for all codes
    code_data = []
    for severity, codes in DRISHTI_CODES.items():
        for code in codes:
            if code in df.columns:
                rate = df[code].mean() * 100
                desc = CODE_DESCRIPTIONS.get(code, code)
                label = code.replace('drishti_', '')
                code_data.append({
                    'code': label,
                    'description': desc,
                    'rate': rate,
                    'severity': severity,
                })

    code_df = pd.DataFrame(code_data).sort_values('rate', ascending=True)

    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    y = np.arange(len(code_df))
    colors = [SEVERITY_COLORS[s] for s in code_df['severity']]

    bars = ax.barh(y, code_df['rate'], color=colors, edgecolor='black',
                   linewidth=0.3, height=0.7, zorder=3)

    # Labels: "P08: Misaligned file"
    labels = [f"{row['code']}: {row['description']}"
              for _, row in code_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel('Trigger Rate (%)')
    ax.set_xlim(0, 100)

    # Highlight excluded codes (P08, M01)
    for i, (_, row) in enumerate(code_df.iterrows()):
        if row['code'] in ('P08', 'M01'):
            ax.get_yticklabels()[i].set_fontweight('bold')
            # Add annotation
            ax.annotate('excluded', xy=(row['rate'], i),
                        xytext=(row['rate'] + 2, i),
                        fontsize=5, color='red', va='center')

    # Legend for severity levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SEVERITY_COLORS['HIGH'], edgecolor='black',
              linewidth=0.3, label='HIGH'),
        Patch(facecolor=SEVERITY_COLORS['WARN'], edgecolor='black',
              linewidth=0.3, label='WARN'),
        Patch(facecolor=SEVERITY_COLORS['INFO'], edgecolor='black',
              linewidth=0.3, label='INFO'),
        Patch(facecolor=SEVERITY_COLORS['OK'], edgecolor='black',
              linewidth=0.3, label='OK'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6,
              title='Severity', title_fontsize=6.5)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'fig_drishti_code_rates')


# ===================================================================
# L4: Confidence Score Histogram
# ===================================================================
def fig_L4_confidence(df):
    """Histogram of Drishti confidence scores."""
    logger.info("Generating L4: Confidence score histogram...")

    if 'drishti_confidence' not in df.columns:
        logger.warning("  drishti_confidence column not found, skipping L4")
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    conf = df['drishti_confidence'].dropna()
    ax.hist(conf, bins=50, color=COLORS['blue'], edgecolor='black',
            linewidth=0.3, alpha=0.85, zorder=3)

    ax.set_xlabel('Drishti Confidence Score')
    ax.set_ylabel('Count')

    # Mark the 0.5 threshold (healthy default)
    ax.axvline(x=0.5, color=COLORS['vermilion'], linestyle='--',
               linewidth=1, alpha=0.8, label='Healthy default (0.5)')

    # Annotate peaks
    median_conf = conf.median()
    ax.axvline(x=median_conf, color=COLORS['green'], linestyle=':',
               linewidth=1, alpha=0.8, label=f'Median ({median_conf:.2f})')

    ax.legend(loc='upper right', fontsize=6.5)

    # Add count annotation
    n_healthy = int((conf == 0.5).sum())
    n_total = len(conf)
    ax.text(0.98, 0.85, f'N = {n_total:,}\nHealthy (0.5): {n_healthy:,}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8))

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'fig_confidence_histogram')


# ===================================================================
# Bonus: L5 — Multi-label cardinality distribution
# ===================================================================
def fig_L5_cardinality(df):
    """Distribution of number of active labels per sample."""
    logger.info("Generating L5: Label cardinality distribution...")

    dims = [d for d in DIMENSION_ORDER if d in df.columns and d != 'healthy']
    # Count active bottleneck dimensions per sample (excluding healthy)
    cardinality = df[dims].sum(axis=1).astype(int)

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    counts = cardinality.value_counts().sort_index()
    x = counts.index
    bars = ax.bar(x, counts.values, color=COLORS['blue'], edgecolor='black',
                  linewidth=0.3, zorder=3)

    # Annotate counts
    for bar, val in zip(bars, counts.values):
        if val > len(df) * 0.01:  # Only annotate if > 1%
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    format_count(val), ha='center', va='bottom', fontsize=6.5)

    ax.set_xlabel('Number of Active Bottleneck Dimensions')
    ax.set_ylabel('Number of Jobs')
    ax.set_xticks(range(int(x.max()) + 1))

    # Add mean cardinality
    mean_card = cardinality.mean()
    ax.axvline(x=mean_card, color=COLORS['vermilion'], linestyle='--',
               linewidth=1, label=f'Mean = {mean_card:.2f}')
    ax.legend(fontsize=6.5)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'fig_label_cardinality')


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate labeling figures for SC 2026 paper'
    )
    parser.add_argument(
        '--figures', nargs='+', default=['L1', 'L2', 'L3', 'L4', 'L5'],
        help='Which figures to generate (default: all)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory',
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    apply_style()
    df = load_heuristic_labels()

    fig_map = {
        'L1': fig_L1_label_distribution,
        'L2': fig_L2_cooccurrence,
        'L3': fig_L3_drishti_codes,
        'L4': fig_L4_confidence,
        'L5': fig_L5_cardinality,
    }

    for fig_id in args.figures:
        fig_id = fig_id.upper()
        if fig_id in fig_map:
            fig_map[fig_id](df)
        else:
            logger.warning(f"Unknown figure: {fig_id}")

    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
