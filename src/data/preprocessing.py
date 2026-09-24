"""
Data Preprocessing Pipeline
============================
Multi-stage pipeline for cleaning, engineering, analyzing, and normalizing
features extracted from Darshan logs.

Pipeline stages (each saves an intermediate parquet):
  Stage 1: Raw extraction (batch_extract.py --raw) -> raw_features.parquet
  Stage 2: Cleaning -> cleaned_features.parquet
  Stage 3: Feature engineering -> features.parquet
  Stage 4: Statistical analysis (EDA) -> stats report (no parquet)
  Stage 5: Normalization -> normalized_features.parquet

Design principles:
  - NO hardcoded feature exclusions.  All exclusion decisions are driven by
    EDA (Stage 4) and configured in preprocessing.yaml.
  - Group-specific normalization: different transforms for different counter
    types (volume, count, histogram, timing, ratio, indicator).
  - Save intermediate stages: Stage 1 is immutable ground truth; Stage 2 is
    the cleaned baseline for alternative normalization experiments.
  - Fit normalizers on training data only; validation and test rows are
    transformed with the fitted scalers.
  - Every stage checks the schema version and the required columns of its
    input; a missing column is an error, never a zero.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.feature_extraction import (
    FEATURE_GROUPS,
    FEATURE_SCHEMA_VERSION,
    DERIVED_FEATURE_NAMES,
    _EPS,
    _SENTINEL,
    compute_layer_and_rank_features,
    extract_raw_features,
    get_raw_feature_names,
)

logger = logging.getLogger(__name__)

_RANK_SENTINEL_COLUMNS = [
    'POSIX_FASTEST_RANK', 'POSIX_FASTEST_RANK_BYTES',
    'POSIX_SLOWEST_RANK', 'POSIX_SLOWEST_RANK_BYTES',
]
_UNAVAILABLE_SENTINEL_COLUMNS = [
    'POSIX_MMAPS', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
]


# ---------------------------------------------------------------------------
# Input contracts
# ---------------------------------------------------------------------------

def require_raw_schema(df, stage):
    """Refuse a frame that was not written by the current extractor.

    Checks the ``_schema_version`` column and every raw feature name, so a
    parquet from an older extraction cannot be engineered into false zeros.
    """
    if '_schema_version' not in df.columns:
        raise ValueError(f"{stage}: input has no _schema_version column; "
                         f"re-extract with feature schema {FEATURE_SCHEMA_VERSION}")
    versions = set(pd.unique(df['_schema_version']))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"{stage}: input schema version(s) {sorted(versions)}, "
                         f"code expects {FEATURE_SCHEMA_VERSION}; re-extract")
    missing = [c for c in get_raw_feature_names() if c not in df.columns]
    if missing:
        raise ValueError(f"{stage}: input lacks {len(missing)} raw columns, "
                         f"first: {missing[:5]}")


def apply_sentinel_handling(df, config):
    """Replace documented Darshan sentinels and reject unexpected ones."""
    sentinel_config = config['sentinel_handling']
    expected_keys = {'replace_negative_rank_with', 'replace_unavailable_counter_with'}
    if set(sentinel_config) != expected_keys:
        raise ValueError(
            "sentinel_handling keys differ from the required contract: "
            f"missing={sorted(expected_keys - set(sentinel_config))}, "
            f"unknown={sorted(set(sentinel_config) - expected_keys)}"
        )
    df = df.copy()
    replacements = {}
    for column in _RANK_SENTINEL_COLUMNS:
        count = int((df[column] == _SENTINEL).sum())
        if count:
            df.loc[df[column] == _SENTINEL, column] = sentinel_config['replace_negative_rank_with']
            replacements[column] = count
    for column in _UNAVAILABLE_SENTINEL_COLUMNS:
        count = int((df[column] == _SENTINEL).sum())
        if count:
            df.loc[df[column] == _SENTINEL, column] = sentinel_config['replace_unavailable_counter_with']
            replacements[column] = count
    known = set(_RANK_SENTINEL_COLUMNS + _UNAVAILABLE_SENTINEL_COLUMNS)
    unexpected = [
        column for column in get_raw_feature_names()
        if column in df.columns and column not in known and (df[column] == _SENTINEL).any()
    ]
    if unexpected:
        raise ValueError(f"unexpected -1 sentinel values in raw columns {unexpected[:5]}")
    return df, replacements


# ---------------------------------------------------------------------------
# Stage 2: Cleaning
# ---------------------------------------------------------------------------

def stage2_clean(
    df: pd.DataFrame, config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean raw features: filter invalid jobs, handle sentinels, add indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features from Stage 1 (``raw_features.parquet``).
    config : dict
        Preprocessing configuration (from ``configs/preprocessing.yaml``).

    Returns
    -------
    pd.DataFrame
        Cleaned features.
    dict
        Cleaning report (counts of removed/modified rows).
    """
    require_raw_schema(df, 'stage2_clean')
    cleaning = config['cleaning']
    report = {'initial_rows': len(df)}

    min_bytes = cleaning['min_total_bytes']
    min_ops = cleaning['min_io_ops']

    # --- Filter: require POSIX module, unless the job's I/O went through
    # stdio alone (fwrite, fprintf, C++ streams never reach POSIX) ---
    if cleaning['require_posix']:
        stdio_bytes = df['STDIO_BYTES_READ'] + df['STDIO_BYTES_WRITTEN']
        mask = (df['has_posix'] == 1) | (stdio_bytes >= min_bytes)
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.info("Removed %d jobs without POSIX module and with less "
                        "than %d STDIO bytes", n_dropped, min_bytes)
            df = df[mask].copy()
    report['after_require_posix'] = len(df)

    # --- Filter: minimum duration ---
    min_duration = cleaning['min_duration_seconds']
    mask = df['runtime_seconds'] >= min_duration
    n_dropped = (~mask).sum()
    logger.info("Removed %d jobs with runtime < %d seconds",
                 n_dropped, min_duration)
    df = df[mask].copy()
    report['after_min_duration'] = len(df)

    # --- Filter: minimum total bytes (POSIX + STDIO) ---
    total_bytes = (df['POSIX_BYTES_READ'] + df['POSIX_BYTES_WRITTEN']
                   + df['STDIO_BYTES_READ'] + df['STDIO_BYTES_WRITTEN'])
    mask = total_bytes >= min_bytes
    n_dropped = (~mask).sum()
    logger.info("Removed %d jobs with total bytes < %d", n_dropped, min_bytes)
    df = df[mask].copy()
    report['after_min_bytes'] = len(df)

    # --- Filter: minimum I/O operations (POSIX + STDIO) ---
    total_ops = (df['POSIX_READS'] + df['POSIX_WRITES']
                 + df['STDIO_READS'] + df['STDIO_WRITES'])
    mask = total_ops >= min_ops
    n_dropped = (~mask).sum()
    logger.info("Removed %d jobs with total ops < %d", n_dropped, min_ops)
    df = df[mask].copy()
    report['after_min_ops'] = len(df)

    # --- Filter: non-negative timing ---
    for col in ['POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME']:
        mask = df[col] >= 0
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.info("Removed %d jobs with negative %s", n_dropped, col)
            df = df[mask].copy()

    report['after_timing_filter'] = len(df)

    # Rows are positions from here on: the split indices of stage 5 index
    # this frame and the parquet written from it by position.
    df = df.reset_index(drop=True)

    # --- Handle documented sentinel values ---
    df, replacements = apply_sentinel_handling(df, config)
    report['sentinel_replacements'] = replacements

    report['final_rows'] = len(df)
    report['rows_removed'] = report['initial_rows'] - report['final_rows']
    report['removal_pct'] = (
        100 * report['rows_removed'] / max(report['initial_rows'], 1)
    )

    logger.info(
        "Cleaning complete: %d -> %d rows (removed %d, %.1f%%)",
        report['initial_rows'], report['final_rows'],
        report['rows_removed'], report['removal_pct']
    )
    return df, report


# ---------------------------------------------------------------------------
# Stage 3: Feature Engineering
# ---------------------------------------------------------------------------

def stage3_engineer(
    df: pd.DataFrame, config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Compute derived features from cleaned raw counters (vectorized).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned features from Stage 2.

    Returns
    -------
    pd.DataFrame
        Features with derived columns added.
    """
    logger.info("Computing derived features for %d rows (vectorized)...",
                len(df))
    require_raw_schema(df, 'stage3_engineer')
    if config is None:
        config = load_preprocessing_config()
    df, _ = apply_sentinel_handling(df, config)

    def g(col):
        # every raw column exists (require_raw_schema); a typo must not become 0
        return df[col]

    # Precompute reusable aggregates
    total_reads = g('POSIX_READS')
    total_writes = g('POSIX_WRITES')
    total_ops = total_reads + total_writes
    bytes_read = g('POSIX_BYTES_READ')
    bytes_written = g('POSIX_BYTES_WRITTEN')
    total_bytes = bytes_read + bytes_written
    read_time = g('POSIX_F_READ_TIME')
    write_time = g('POSIX_F_WRITE_TIME')
    meta_time = g('POSIX_F_META_TIME')
    total_time = read_time + write_time + meta_time
    nprocs = df['nprocs']
    runtime = df['runtime_seconds']

    # --- Read/write balance ---
    df['read_ratio'] = bytes_read / np.maximum(total_bytes, 1)

    # --- Bandwidth ---
    df['read_bw_mb_s'] = bytes_read / np.maximum(read_time, _EPS) / 1e6
    df['write_bw_mb_s'] = bytes_written / np.maximum(write_time, _EPS) / 1e6
    io_time = read_time + write_time
    df['total_bw_mb_s'] = total_bytes / np.maximum(io_time, _EPS) / 1e6

    # --- Average sizes ---
    df['avg_read_size'] = bytes_read / np.maximum(total_reads, 1)
    df['avg_write_size'] = bytes_written / np.maximum(total_writes, 1)

    # --- Size distribution ratios ---
    small_r = g('POSIX_SIZE_READ_0_100') + g('POSIX_SIZE_READ_100_1K')
    small_w = g('POSIX_SIZE_WRITE_0_100') + g('POSIX_SIZE_WRITE_100_1K')
    df['small_read_ratio'] = small_r / np.maximum(total_reads, 1)
    df['small_write_ratio'] = small_w / np.maximum(total_writes, 1)
    df['small_io_ratio'] = (small_r + small_w) / np.maximum(total_ops, 1)

    medium_r = (g('POSIX_SIZE_READ_1K_10K') + g('POSIX_SIZE_READ_10K_100K')
                + g('POSIX_SIZE_READ_100K_1M'))
    medium_w = (g('POSIX_SIZE_WRITE_1K_10K') + g('POSIX_SIZE_WRITE_10K_100K')
                + g('POSIX_SIZE_WRITE_100K_1M'))
    df['medium_read_ratio'] = medium_r / np.maximum(total_reads, 1)
    df['medium_write_ratio'] = medium_w / np.maximum(total_writes, 1)

    large_r = (g('POSIX_SIZE_READ_1M_4M') + g('POSIX_SIZE_READ_4M_10M')
               + g('POSIX_SIZE_READ_10M_100M') + g('POSIX_SIZE_READ_100M_1G')
               + g('POSIX_SIZE_READ_1G_PLUS'))
    large_w = (g('POSIX_SIZE_WRITE_1M_4M') + g('POSIX_SIZE_WRITE_4M_10M')
               + g('POSIX_SIZE_WRITE_10M_100M') + g('POSIX_SIZE_WRITE_100M_1G')
               + g('POSIX_SIZE_WRITE_1G_PLUS'))
    df['large_read_ratio'] = large_r / np.maximum(total_reads, 1)
    df['large_write_ratio'] = large_w / np.maximum(total_writes, 1)

    # --- Pattern ratios ---
    df['seq_read_ratio'] = g('POSIX_SEQ_READS') / np.maximum(total_reads, 1)
    df['seq_write_ratio'] = g('POSIX_SEQ_WRITES') / np.maximum(total_writes, 1)
    df['consec_read_ratio'] = g('POSIX_CONSEC_READS') / np.maximum(total_reads, 1)
    df['consec_write_ratio'] = g('POSIX_CONSEC_WRITES') / np.maximum(total_writes, 1)
    df['rw_ratio'] = total_reads / np.maximum(total_writes, 1)
    df['rw_switch_ratio'] = g('POSIX_RW_SWITCHES') / np.maximum(total_ops, 1)

    # --- Alignment ratios ---
    df['mem_misalign_ratio'] = g('POSIX_MEM_NOT_ALIGNED') / np.maximum(total_ops, 1)
    df['file_misalign_ratio'] = g('POSIX_FILE_NOT_ALIGNED') / np.maximum(total_ops, 1)

    # --- Metadata ratios ---
    df['metadata_time_ratio'] = meta_time / np.maximum(total_time, _EPS)
    df['read_time_fraction'] = read_time / np.maximum(total_time, _EPS)
    df['write_time_fraction'] = write_time / np.maximum(total_time, _EPS)
    df['opens_per_op'] = g('POSIX_OPENS') / np.maximum(total_ops, 1)
    df['stats_per_op'] = g('POSIX_STATS') / np.maximum(total_ops, 1)
    df['seeks_per_op'] = g('POSIX_SEEKS') / np.maximum(total_ops, 1)
    df['fsync_ratio'] = g('POSIX_FSYNCS') / np.maximum(total_writes, 1)
    df['opens_per_mb'] = g('POSIX_OPENS') / np.maximum(total_bytes / 1e6, _EPS)

    # --- Imbalance ratios ---
    var_bytes = np.maximum(g('POSIX_F_VARIANCE_RANK_BYTES'), 0)
    var_time = np.maximum(g('POSIX_F_VARIANCE_RANK_TIME'), 0)
    mean_bytes_per_rank = total_bytes / np.maximum(nprocs, 1)
    mean_time_per_rank = total_time / np.maximum(nprocs, 1)
    df['rank_bytes_cv'] = np.sqrt(var_bytes) / np.maximum(mean_bytes_per_rank, _EPS)
    df['rank_time_cv'] = np.sqrt(var_time) / np.maximum(mean_time_per_rank, _EPS)

    fastest_bytes = g('POSIX_FASTEST_RANK_BYTES')
    slowest_bytes = g('POSIX_SLOWEST_RANK_BYTES')
    df['byte_imbalance'] = (slowest_bytes - fastest_bytes) / np.maximum(total_bytes, _EPS)

    fastest_time = g('POSIX_F_FASTEST_RANK_TIME')
    slowest_time = g('POSIX_F_SLOWEST_RANK_TIME')
    df['time_imbalance'] = (slowest_time - fastest_time) / np.maximum(total_time, _EPS)

    # --- MPI-IO ratios ---
    coll = g('MPIIO_COLL_READS') + g('MPIIO_COLL_WRITES')
    indep = g('MPIIO_INDEP_READS') + g('MPIIO_INDEP_WRITES')
    nb = g('MPIIO_NB_READS') + g('MPIIO_NB_WRITES')
    total_mpiio = coll + indep + nb
    df['collective_ratio'] = coll / np.maximum(total_mpiio, 1)
    df['nonblocking_ratio'] = nb / np.maximum(total_mpiio, 1)

    # --- Temporal ---
    open_start = g('POSIX_F_OPEN_START_TIMESTAMP')
    close_end = g('POSIX_F_CLOSE_END_TIMESTAMP')
    df['io_duration'] = np.maximum(close_end - open_start, 0)
    df['io_active_fraction'] = total_time / np.maximum(runtime, _EPS)

    # --- Access concentration ---
    df['access_size_concentration'] = g('POSIX_ACCESS1_COUNT') / np.maximum(total_ops, 1)
    df['dominant_access_size'] = g('POSIX_ACCESS1_ACCESS')

    # --- Layer-agnostic totals and per-rank distribution ---
    for name, values in compute_layer_and_rank_features(g, nprocs).items():
        df[name] = values

    n_derived = len(DERIVED_FEATURE_NAMES)
    logger.info("Added %d derived features (total: %d columns)",
                n_derived, len(df.columns))
    return df


def engineer_one(
    parsed_log: dict[str, object], config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Raw plus derived features of one parsed log, as a dict.

    The same two steps the batch pipeline runs (``extract_raw_features``,
    then ``stage3_engineer``), for callers that handle single logs.
    """
    df = stage3_engineer(pd.DataFrame([extract_raw_features(parsed_log)]), config=config)
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Stage 4: Statistical Analysis (EDA)
# ---------------------------------------------------------------------------

def compute_statistics(df):
    """Compute the statistics used for EDA.

    Run this on Stage 2 (cleaned) or Stage 3 (engineered) features to
    inform decisions about feature exclusion and normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (cleaned or engineered).

    Returns
    -------
    pd.DataFrame
        Statistics per feature: count, mean, std, min, max, percentiles,
        skewness, kurtosis, zero_fraction, negative_fraction.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])

    stats = pd.DataFrame(index=numeric_df.columns)
    stats['count'] = numeric_df.count()
    stats['mean'] = numeric_df.mean()
    stats['std'] = numeric_df.std()
    stats['min'] = numeric_df.min()
    stats['p01'] = numeric_df.quantile(0.01)
    stats['p05'] = numeric_df.quantile(0.05)
    stats['p25'] = numeric_df.quantile(0.25)
    stats['median'] = numeric_df.median()
    stats['p75'] = numeric_df.quantile(0.75)
    stats['p95'] = numeric_df.quantile(0.95)
    stats['p99'] = numeric_df.quantile(0.99)
    stats['max'] = numeric_df.max()
    stats['skewness'] = numeric_df.skew()
    stats['kurtosis'] = numeric_df.kurtosis()
    stats['zero_fraction'] = (numeric_df == 0).mean()
    stats['negative_fraction'] = (numeric_df < 0).mean()
    stats['nonzero_count'] = (numeric_df != 0).sum()

    # Identify feature group for each column
    group_map = {}
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            group_map[col] = group_name
    stats['feature_group'] = stats.index.map(
        lambda x: group_map.get(x, 'other')
    )

    logger.info("Computed statistics for %d features", len(stats))
    return stats


def compute_correlation_matrix(df, method='spearman'):
    """Compute feature correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    method : str
        Correlation method ('spearman' recommended for heavy-tailed data).

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def find_redundant_features(corr_matrix, threshold=0.90):
    """Find pairs of features with correlation above threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix from ``compute_correlation_matrix``.
    threshold : float
        Absolute correlation threshold.

    Returns
    -------
    list[tuple]
        List of (feature_a, feature_b, correlation) tuples.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                pairs.append((cols[i], cols[j], corr_val))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Feature Exclusion (between EDA and Normalization)
# ---------------------------------------------------------------------------

def drop_excluded_features(df, config, train_df=None):
    """Drop features that should not go to ML models.

    Two mechanisms:
      1. Auto-drop: constant features (zero variance) detected on train_df.
      2. Manual exclusion: features listed in config['feature_exclusion'].

    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    config : dict
        Preprocessing config with 'feature_exclusion' section.
    train_df : pd.DataFrame, optional
        Training set for detecting constant features.  If None, uses df.

    Returns
    -------
    pd.DataFrame
        DataFrame with excluded features removed.
    list
        Names of dropped features (for logging/auditing).
    """
    exclusion_cfg = config.get('feature_exclusion', {})
    dropped = []

    # 1. Manual exclusions from config
    manual_drops = exclusion_cfg.get('drop_features', [])
    present = [c for c in manual_drops if c in df.columns]
    if present:
        df = df.drop(columns=present)
        dropped.extend(present)
        logger.info("Dropped %d manually excluded features: %s",
                     len(present), present[:5])

    # 2. Auto-drop constant features (zero variance on train set)
    if exclusion_cfg.get('drop_constant', True):
        ref = train_df if train_df is not None else df
        feature_cols = [c for c in ref.columns
                        if not c.startswith('_') and c in df.columns]
        numeric = ref[feature_cols].select_dtypes(include=['number'])
        constant_cols = numeric.columns[numeric.std() == 0].tolist()

        # Only drop those still present (some may already be manually dropped)
        to_drop = [c for c in constant_cols if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            dropped.extend(to_drop)
            logger.info("Auto-dropped %d constant features: %s",
                         len(to_drop), to_drop[:5])

    logger.info("Feature exclusion complete: dropped %d features, %d remain",
                len(dropped), len([c for c in df.columns if not c.startswith('_')]))
    return df, dropped


# ---------------------------------------------------------------------------
# Stage 5: Normalization
# ---------------------------------------------------------------------------

def stage5_normalize(
    df: pd.DataFrame,
    config: dict[str, object],
    fit: bool = True,
    scalers: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply group-specific normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features (Stage 3 output).
    config : dict
        Preprocessing configuration with normalization settings.
    fit : bool
        If True, fit scalers on this data (training set).
        If False, use pre-fitted scalers (validation/test set).
    scalers : dict, optional
        Pre-fitted scalers keyed by group name.  Required when fit=False.

    Returns
    -------
    pd.DataFrame
        Normalized features.
    dict
        Fitted scalers (save for inference on validation/test).
    """
    from sklearn.preprocessing import RobustScaler

    norm_config = config['normalization']
    if scalers is None:
        scalers = {}

    df = df.copy()
    feature_cols = [c for c in df.columns if not c.startswith('_')]

    config_keys = {
        'volume': 'volume_counters', 'count': 'count_counters',
        'histogram': 'histogram_counters', 'top4': 'top4_counters',
        'timing': 'timing_counters', 'timestamp': 'timestamp_counters',
        'categorical': 'categorical_counters', 'rank_id': 'rank_id_counters',
        'rank_stat': 'rank_stat_counters',
        'rank_stat_bounded': 'rank_stat_bounded_counters',
        'conditional_size': 'conditional_size_counters',
        'indicator': 'indicator_features', 'ratio': 'ratio_features',
        'ratio_unbounded': 'ratio_unbounded_features',
        'derived_absolute': 'derived_absolute', 'metadata': 'metadata_features',
    }
    expected_keys = set(config_keys.values())
    if set(norm_config) != expected_keys:
        raise ValueError(
            "normalization keys differ from the required contract: "
            f"missing={sorted(expected_keys - set(norm_config))}, "
            f"unknown={sorted(set(norm_config) - expected_keys)}"
        )
    allowed_methods = {'none', 'log1p', 'log1p_robust', 'log10p1'}
    invalid = {key: value for key, value in norm_config.items()
               if value not in allowed_methods}
    if invalid:
        raise ValueError(f"unknown normalization methods: {invalid}")
    group_methods = {
        group: norm_config[config_key] for group, config_key in config_keys.items()
    }

    # Apply group-specific normalization (vectorized)
    for group_name, method in group_methods.items():
        cols = [c for c in FEATURE_GROUPS.get(group_name, [])
                if c in df.columns]
        if not cols or method == 'none':
            continue

        if method == 'log1p':
            df[cols] = np.log1p(df[cols].clip(lower=0))

        elif method == 'log1p_robust':
            # Step 1: log1p transform (vectorized)
            df[cols] = np.log1p(df[cols].clip(lower=0))

            # Step 2: RobustScaler (median + IQR)
            if fit:
                scaler = RobustScaler()
                df[cols] = scaler.fit_transform(df[cols])
                scalers[group_name] = scaler
                logger.info("Fitted RobustScaler for %s (%d features)",
                            group_name, len(cols))
            else:
                if group_name not in scalers:
                    raise ValueError(f"no fitted scaler for group {group_name}; "
                                     "fit on the training split first")
                df[cols] = scalers[group_name].transform(df[cols])

        elif method == 'log10p1':
            # Legacy: log10(x+1) for backward compatibility
            df[cols] = np.log10(df[cols].clip(lower=0) + 1)

        else:
            raise AssertionError(f"normalization method was not validated: {method}")

    logger.info("Normalization complete: %d columns", len(feature_cols))
    return df, scalers


# ---------------------------------------------------------------------------
# Data Splits
# ---------------------------------------------------------------------------

def create_splits(df, config):
    """Create job-grouped train/val/test splits as row positions of ``df``.

    ``splits.method`` is ``temporal`` (sort by ``_start_time``, the oldest
    rows train and the newest test) or ``random`` (seeded shuffle). Both
    return positions, so consumers index the frame, or a parquet written
    from it, with ``iloc``. Rows from one ``(_uid, _jobid)`` group stay in one
    partition. The three partitions are checked to be disjoint, to cover
    every row, and to be non-empty.

    Returns
    -------
    dict
        ``{'train_idx': array, 'val_idx': array, 'test_idx': array}``
    """
    split_config = config['splits']
    method = split_config['method']
    test_fraction = split_config['test_fraction']
    val_fraction = split_config['val_fraction']
    seed = config['random_seed']

    if not (0 < test_fraction < 1 and 0 < val_fraction < 1
            and test_fraction + val_fraction < 1):
        raise ValueError(f"split fractions must be in (0, 1) and sum below 1: "
                         f"test={test_fraction}, val={val_fraction}")
    n = len(df)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    if n_test == 0 or n_val == 0 or n - n_test - n_val == 0:
        raise ValueError(f"{n} rows give an empty partition at fractions "
                         f"test={test_fraction}, val={val_fraction}")

    required = ['_uid', '_jobid', '_start_time']
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"grouped split lacks columns {missing}")
    group_rows = {}
    row_keys = []
    for pos, (uid, jobid) in enumerate(zip(df['_uid'], df['_jobid'])):
        if jobid != 0:
            key = (uid, jobid)
        elif '_source_path' in df.columns:
            source_path = df['_source_path'].iloc[pos]
            if pd.isna(source_path) or not str(source_path).strip():
                raise ValueError(
                    "grouped split cannot identify a row with _jobid 0 and an empty _source_path"
                )
            key = ('path', str(source_path))
        else:
            raise ValueError(
                "grouped split cannot identify a row with _jobid 0 and no _source_path"
            )
        group_rows.setdefault(key, []).append(pos)
        row_keys.append(key)
    if len(group_rows) < 3:
        raise ValueError("grouped split needs at least three job groups")

    if method == 'temporal':
        group_order = sorted(group_rows, key=lambda key: (
            min(df['_start_time'].iloc[group_rows[key]]), group_rows[key][0]))
    elif method == 'random':
        group_order = list(group_rows)
        np.random.RandomState(seed).shuffle(group_order)
    else:
        raise ValueError(f"unknown split method {method!r}")

    sizes = np.array([len(group_rows[key]) for key in group_order])
    cumulative = np.cumsum(sizes)
    train_target = n - n_test - n_val
    train_cut = min(range(1, len(group_order) - 1),
                    key=lambda cut: abs(cumulative[cut - 1] - train_target))
    val_target_end = n - n_test
    val_cut = min(range(train_cut + 1, len(group_order)),
                  key=lambda cut: abs(cumulative[cut - 1] - val_target_end))
    ordered_rows = [np.asarray(group_rows[key], dtype=int) for key in group_order]
    splits = {
        'train_idx': np.concatenate(ordered_rows[:train_cut]),
        'val_idx': np.concatenate(ordered_rows[train_cut:val_cut]),
        'test_idx': np.concatenate(ordered_rows[val_cut:]),
    }
    parts = np.concatenate(list(splits.values()))
    if len(np.unique(parts)) != n or parts.min() != 0 or parts.max() != n - 1:
        raise AssertionError("split partitions must be disjoint and cover every row")
    keys_by_part = []
    for idx in splits.values():
        keys_by_part.append({row_keys[pos] for pos in idx})
    if any(keys_by_part[i] & keys_by_part[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("a production job appears in more than one partition")
    logger.info("%s grouped split: train=%d, val=%d, test=%d (%d jobs)",
                method.capitalize(), *(len(idx) for idx in splits.values()), len(group_rows))
    return splits


# ---------------------------------------------------------------------------
# Sparse feature detection (for EDA, not automatic removal)
# ---------------------------------------------------------------------------

def find_sparse_features(df, max_zero_fraction=0.99):
    """Identify features with high zero fraction (candidates for removal).

    This is an analysis function for EDA, not an automatic removal step.
    The final exclusion decision should be made after reviewing statistics
    and domain relevance.

    Parameters
    ----------
    df : pd.DataFrame
    max_zero_fraction : float
        Features with > this fraction of zeros are flagged.

    Returns
    -------
    list[tuple]
        List of (column_name, zero_fraction) for flagged features.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    sparse = []
    for col in feature_cols:
        zero_frac = (df[col] == 0).mean()
        if zero_frac > max_zero_fraction:
            sparse.append((col, zero_frac))
    sparse.sort(key=lambda x: x[1], reverse=True)
    return sparse


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def load_preprocessing_config(config_path=None):
    """Load preprocessing configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to preprocessing.yaml.  Defaults to
        ``configs/preprocessing.yaml`` relative to project root.

    Returns
    -------
    dict
        Configuration dictionary. A missing file is an error: the thresholds
        decide which jobs enter the dataset, so there is no built-in default.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / 'configs' / 'preprocessing.yaml'
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"preprocessing config not found: {config_path}")
    with open(config_path) as fh:
        return yaml.safe_load(fh)
