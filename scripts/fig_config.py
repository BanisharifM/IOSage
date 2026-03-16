"""
Shared figure configuration for SC 2026 paper.
===============================================
All figure scripts import from here to ensure consistent styling.
See docs/figure_style_guide.md for rationale.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# IEEE two-column figure widths
SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.0, 2.8)
DOUBLE_COL_TALL = (7.0, 4.5)
TRIPLE_PANEL = (7.0, 2.4)

# Okabe-Ito colorblind-safe palette (8 colors)
PALETTE_8 = [
    '#0072B2',  # blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#D55E00',  # vermilion
    '#CC79A7',  # pink
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#BBBBBB',  # gray
]

COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermilion': '#D55E00',
    'pink': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'gray': '#BBBBBB',
    'black': '#000000',
}

# Hatching patterns for B&W readability
HATCHES = ['', '//', '\\\\', 'xx', '..', '++', 'oo', '**']

# I/O health dimension labels (consistent across all figures)
DIMENSION_ORDER = [
    'access_granularity', 'metadata_intensity', 'parallelism_efficiency',
    'access_pattern', 'interface_choice', 'file_strategy',
    'throughput_utilization', 'healthy',
]

DIMENSION_LABELS = {
    'access_granularity':     'Access\nGranularity',
    'metadata_intensity':     'Metadata\nIntensity',
    'parallelism_efficiency': 'Parallelism\nEfficiency',
    'access_pattern':         'Access\nPattern',
    'interface_choice':       'Interface\nChoice',
    'file_strategy':          'File\nStrategy',
    'throughput_utilization':  'Throughput\nUtilization',
    'healthy':                'Healthy',
}

DIMENSION_LABELS_SHORT = {
    'access_granularity':     'Acc. Gran.',
    'metadata_intensity':     'Meta. Int.',
    'parallelism_efficiency': 'Par. Eff.',
    'access_pattern':         'Acc. Pat.',
    'interface_choice':       'Iface. Ch.',
    'file_strategy':          'File Str.',
    'throughput_utilization':  'Thr. Util.',
    'healthy':                'Healthy',
}


def apply_style():
    """Apply IEEE-compliant publication style to matplotlib."""
    plt.rcParams.update({
        # Fonts — serif to match IEEE body text
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',

        # Axes
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Ticks
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.handlelength': 1.5,

        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 4,

        # Font embedding (CRITICAL for IEEE)
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def save_figure(fig, output_dir, name):
    """Save figure as PDF (vector) and PNG (300 DPI)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f'{name}.pdf'
    png_path = output_dir / f'{name}.png'

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png')
    plt.close(fig)
    logger.info(f"  Saved: {pdf_path} and {png_path}")


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add (a), (b), (c) panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom')


def format_count(n):
    """Format large numbers: 40250 -> '40.3K'."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)
