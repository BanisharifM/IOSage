"""
Preprocessing Pipeline Runner
==============================
Orchestrates Stages 2-5 of the preprocessing pipeline on raw_features.parquet.

Stages:
  2. Cleaning        -> data/processed/production/cleaned_features.parquet
  3. Engineering      -> data/processed/production/features.parquet
  4. EDA / Statistics -> data/processed/production/eda/stats.parquet + eda_report.json
  5. Normalization    -> data/processed/normalized_{train,val,test}.parquet
                      + data/processed/scalers.pkl

Usage::

    python scripts/run_preprocessing.py \
        --input data/processed/raw_features.parquet \
        --output-dir data/processed \
        --config configs/preprocessing.yaml

    # Resume from a specific stage (skip earlier stages if outputs exist)
    python scripts/run_preprocessing.py \
        --input data/processed/raw_features.parquet \
        --output-dir data/processed \
        --start-stage 3

    # Quick test on sample
    python scripts/run_preprocessing.py \
        --input data/processed/raw_features.parquet \
        --output-dir data/processed \
        --sample 10000
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    compute_correlation_matrix,
    compute_statistics,
    create_splits,
    drop_excluded_features,
    find_redundant_features,
    find_sparse_features,
    load_preprocessing_config,
    stage2_clean,
    stage3_engineer,
    stage5_normalize,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir, level=logging.INFO):
    """Configure logging to both console and file."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'preprocessing_{timestamp}.log'

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root.addHandler(fh)

    logger.info("Logging to %s", log_file)
    return log_file


def validate_dataframe(df, stage_name, expected_min_rows=1000):
    """Run basic validation checks on a DataFrame after a stage."""
    issues = []
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

    # Check for NaN
    nan_counts = df[numeric_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        issues.append(f"NaN found in {len(nan_cols)} columns: "
                      f"{list(nan_cols.head(5).index)}")

    # Check for Inf
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        issues.append(f"Inf found in {len(inf_cols)} columns: "
                      f"{list(inf_cols.head(5).index)}")

    # Check row count
    if len(df) < expected_min_rows:
        issues.append(f"Only {len(df)} rows (expected >= {expected_min_rows})")

    # Check for all-zero feature columns
    all_zero = (df[numeric_cols] == 0).all()
    n_all_zero = all_zero.sum()

    if issues:
        for issue in issues:
            logger.warning("[%s] VALIDATION: %s", stage_name, issue)
    else:
        logger.info("[%s] VALIDATION PASSED: %d rows, %d columns, "
                    "%d all-zero features, no NaN/Inf",
                    stage_name, len(df), len(df.columns), n_all_zero)

    return len(issues) == 0


def run_stage2(input_path, output_dir, config):
    """Stage 2: Cleaning."""
    logger.info("=" * 60)
    logger.info("STAGE 2: CLEANING")
    logger.info("=" * 60)

    t0 = time.time()
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows x %d columns from %s",
                len(df), len(df.columns), input_path)

    df_clean, report = stage2_clean(df, config)

    # Save
    out_path = output_dir / 'cleaned_features.parquet'
    df_clean.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    logger.info("Stage 2 complete in %.1fs: %d -> %d rows (removed %d, %.1f%%)",
                elapsed, report['initial_rows'], report['final_rows'],
                report['rows_removed'], report['removal_pct'])
    logger.info("Saved: %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)

    # Log detailed report
    for key, val in report.items():
        logger.info("  %s: %s", key, val)

    validate_dataframe(df_clean, "Stage 2", expected_min_rows=100000)
    return df_clean, report


def run_stage3(df_or_path, output_dir):
    """Stage 3: Feature Engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    t0 = time.time()
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_parquet(df_or_path)
        logger.info("Loaded %d rows from %s", len(df), df_or_path)
    else:
        df = df_or_path

    n_before = len(df.columns)
    df_eng = stage3_engineer(df)
    n_after = len(df_eng.columns)

    # Save
    out_path = output_dir / 'engineered_features.parquet'
    df_eng.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    logger.info("Stage 3 complete in %.1fs: %d -> %d columns (+%d derived)",
                elapsed, n_before, n_after, n_after - n_before)
    logger.info("Saved: %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)

    validate_dataframe(df_eng, "Stage 3", expected_min_rows=100000)
    return df_eng


def run_stage4(df_or_path, output_dir, config):
    """Stage 4: Statistical Analysis (EDA)."""
    logger.info("=" * 60)
    logger.info("STAGE 4: STATISTICAL ANALYSIS (EDA)")
    logger.info("=" * 60)

    t0 = time.time()
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_parquet(df_or_path)
        logger.info("Loaded %d rows from %s", len(df), df_or_path)
    else:
        df = df_or_path

    # Per-feature statistics
    logger.info("Computing per-feature statistics...")
    stats = compute_statistics(df)
    stats_path = output_dir / 'eda_stats.parquet'
    stats.to_parquet(stats_path)
    logger.info("Saved feature statistics: %s (%d features)", stats_path,
                len(stats))

    # Correlation matrix (Spearman — handles non-linear monotonic relationships)
    logger.info("Computing Spearman correlation matrix...")
    feature_sel = config.get('feature_selection', {})
    corr_threshold = feature_sel.get('correlation_threshold', 0.90)
    min_nonzero = feature_sel.get('min_nonzero_fraction', 0.01)

    corr_matrix = compute_correlation_matrix(df, method='spearman')
    corr_path = output_dir / 'eda_correlation.parquet'
    corr_matrix.to_parquet(corr_path)

    # Redundant features
    redundant_pairs = find_redundant_features(corr_matrix, threshold=corr_threshold)
    logger.info("Found %d redundant feature pairs (|rho| > %.2f)",
                len(redundant_pairs), corr_threshold)
    if redundant_pairs:
        for a, b, r in redundant_pairs[:10]:
            logger.info("  %s <-> %s: %.3f", a, b, r)
        if len(redundant_pairs) > 10:
            logger.info("  ... and %d more pairs", len(redundant_pairs) - 10)

    # Sparse features
    sparse_features = find_sparse_features(df, max_zero_fraction=1.0 - min_nonzero)
    logger.info("Found %d sparse features (>%.0f%% zeros)",
                len(sparse_features), (1.0 - min_nonzero) * 100)
    if sparse_features:
        for col, frac in sparse_features[:10]:
            logger.info("  %s: %.1f%% zeros", col, frac * 100)

    # Distribution summary by group
    logger.info("\nFeature group summary:")
    for group in stats['feature_group'].unique():
        group_stats = stats[stats['feature_group'] == group]
        logger.info("  %s (%d features): median_skew=%.1f, "
                    "median_zero_frac=%.2f",
                    group, len(group_stats),
                    group_stats['skewness'].median(),
                    group_stats['zero_fraction'].median())

    # Save EDA report
    eda_report = {
        'n_features': len(stats),
        'n_redundant_pairs': len(redundant_pairs),
        'n_sparse_features': len(sparse_features),
        'redundant_pairs_top20': [(a, b, float(r))
                                   for a, b, r in redundant_pairs[:20]],
        'sparse_features_top20': [(col, float(frac))
                                   for col, frac in sparse_features[:20]],
        'group_summary': {
            group: {
                'n_features': int(len(group_stats)),
                'median_skewness': float(group_stats['skewness'].median()),
                'median_kurtosis': float(group_stats['kurtosis'].median()),
                'median_zero_fraction': float(
                    group_stats['zero_fraction'].median()),
            }
            for group in stats['feature_group'].unique()
            for group_stats in [stats[stats['feature_group'] == group]]
        }
    }
    report_path = output_dir / 'eda_report.json'
    with open(report_path, 'w') as fh:
        json.dump(eda_report, fh, indent=2)

    elapsed = time.time() - t0
    logger.info("Stage 4 complete in %.1fs", elapsed)
    logger.info("Saved: %s, %s, %s",
                stats_path.name, corr_path.name, report_path.name)

    return stats, eda_report


def run_stage5(df_or_path, output_dir, config):
    """Stage 5: Normalization + Splits."""
    logger.info("=" * 60)
    logger.info("STAGE 5: NORMALIZATION + SPLITS")
    logger.info("=" * 60)

    t0 = time.time()
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_parquet(df_or_path)
        logger.info("Loaded %d rows from %s", len(df), df_or_path)
    else:
        df = df_or_path

    # Create splits BEFORE normalization (to prevent data leakage)
    logger.info("Creating train/val/test splits...")
    splits = create_splits(df, config)
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(train_idx), len(val_idx), len(test_idx))

    # Drop excluded features (constant + manually listed in config)
    # Use train set as reference for detecting constant features
    df_train_ref = df.loc[train_idx]
    n_before = len([c for c in df.columns if not c.startswith('_')])
    df, dropped_features = drop_excluded_features(
        df, config, train_df=df_train_ref)
    n_after = len([c for c in df.columns if not c.startswith('_')])
    logger.info("Feature exclusion: %d -> %d features (dropped %d)",
                n_before, n_after, len(dropped_features))

    # Save dropped feature list for reference
    dropped_path = output_dir / 'dropped_features.json'
    import json as _json
    with open(dropped_path, 'w') as fh:
        _json.dump({
            'dropped': dropped_features,
            'count': len(dropped_features),
            'remaining': n_after,
        }, fh, indent=2)

    # Normalize TRAINING set (fit scalers)
    logger.info("Normalizing training set (fitting scalers)...")
    df_train = df.loc[train_idx].copy()
    df_train_norm, scalers = stage5_normalize(df_train, config, fit=True)

    # Normalize VAL and TEST with pre-fitted scalers
    logger.info("Normalizing validation set (transform only)...")
    df_val = df.loc[val_idx].copy()
    df_val_norm, _ = stage5_normalize(df_val, config, fit=False, scalers=scalers)

    logger.info("Normalizing test set (transform only)...")
    df_test = df.loc[test_idx].copy()
    df_test_norm, _ = stage5_normalize(df_test, config, fit=False, scalers=scalers)

    # Save splits
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    df_train_norm.to_parquet(splits_dir / 'train.parquet', index=False)
    df_val_norm.to_parquet(splits_dir / 'val.parquet', index=False)
    df_test_norm.to_parquet(splits_dir / 'test.parquet', index=False)

    # Save scalers
    scaler_path = output_dir / 'scalers.pkl'
    with open(scaler_path, 'wb') as fh:
        pickle.dump(scalers, fh)

    # Save split indices
    split_path = output_dir / 'split_indices.pkl'
    with open(split_path, 'wb') as fh:
        pickle.dump(splits, fh)

    # Also save the full normalized dataset (train scalers applied to all)
    logger.info("Creating full normalized dataset...")
    df_full_norm, _ = stage5_normalize(
        df.copy(), config, fit=False, scalers=scalers)
    df_full_norm.to_parquet(output_dir / 'normalized_features.parquet',
                            index=False)

    elapsed = time.time() - t0
    logger.info("Stage 5 complete in %.1fs", elapsed)

    # Validation
    for name, df_norm in [('train', df_train_norm), ('val', df_val_norm),
                          ('test', df_test_norm)]:
        validate_dataframe(df_norm, f"Stage 5 ({name})",
                          expected_min_rows=1000)

    # Summary statistics of normalized features
    feature_cols = [c for c in df_train_norm.columns if not c.startswith('_')]
    numeric = df_train_norm[feature_cols].select_dtypes(include=[np.number])
    logger.info("\nNormalized train set distribution summary:")
    logger.info("  Mean of means: %.3f", numeric.mean().mean())
    logger.info("  Mean of stds:  %.3f", numeric.std().mean())
    logger.info("  Min value:     %.3f", numeric.min().min())
    logger.info("  Max value:     %.3f", numeric.max().max())
    logger.info("  NaN count:     %d", numeric.isna().sum().sum())

    # Report split file sizes
    for name in ['train', 'val', 'test']:
        p = splits_dir / f'{name}.parquet'
        logger.info("  %s: %d rows, %.1f MB",
                    name, len(splits[f'{name}_idx']),
                    p.stat().st_size / 1e6)

    return scalers, splits


def main():
    parser = argparse.ArgumentParser(
        description='Run preprocessing pipeline (Stages 2-5)')
    parser.add_argument('--input', type=str,
                        default='data/processed/production/raw_features.parquet',
                        help='Input parquet from Stage 1')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for all stages')
    parser.add_argument('--config', type=str,
                        default='configs/preprocessing.yaml',
                        help='Preprocessing configuration YAML')
    parser.add_argument('--start-stage', type=int, default=2,
                        help='Start from this stage (2-5)')
    parser.add_argument('--end-stage', type=int, default=5,
                        help='Stop after this stage (2-5)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N rows for testing')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("Input:      %s", args.input)
    logger.info("Output dir: %s", output_dir)
    logger.info("Config:     %s", args.config)
    logger.info("Stages:     %d to %d", args.start_stage, args.end_stage)
    if args.sample:
        logger.info("Sample:     %d rows", args.sample)

    config = load_preprocessing_config(args.config)
    logger.info("Config loaded: %s", args.config)

    t_total = time.time()

    # Determine starting data
    df = None

    # --- Stage 2: Cleaning ---
    if args.start_stage <= 2 <= args.end_stage:
        df_clean, clean_report = run_stage2(
            Path(args.input), output_dir, config)
        if args.sample and len(df_clean) > args.sample:
            rng = np.random.RandomState(config.get('random_seed', 42))
            idx = rng.choice(len(df_clean), args.sample, replace=False)
            df_clean = df_clean.iloc[idx].reset_index(drop=True)
            logger.info("Sampled %d rows for testing", args.sample)
        df = df_clean
    elif args.start_stage > 2:
        # Load from previous stage output
        cleaned_path = output_dir / 'cleaned_features.parquet'
        if cleaned_path.exists():
            df = pd.read_parquet(cleaned_path)
            logger.info("Loaded cleaned features: %d rows", len(df))
            if args.sample and len(df) > args.sample:
                rng = np.random.RandomState(config.get('random_seed', 42))
                idx = rng.choice(len(df), args.sample, replace=False)
                df = df.iloc[idx].reset_index(drop=True)
                logger.info("Sampled %d rows for testing", args.sample)

    # --- Stage 3: Feature Engineering ---
    if args.start_stage <= 3 <= args.end_stage:
        if df is not None:
            df = run_stage3(df, output_dir)
        else:
            df = run_stage3(output_dir / 'cleaned_features.parquet',
                           output_dir)
    elif args.start_stage > 3:
        eng_path = output_dir / 'engineered_features.parquet'
        if eng_path.exists():
            df = pd.read_parquet(eng_path)
            logger.info("Loaded engineered features: %d rows", len(df))

    # --- Stage 4: EDA ---
    if args.start_stage <= 4 <= args.end_stage:
        if df is not None:
            run_stage4(df, output_dir, config)
        else:
            run_stage4(output_dir / 'engineered_features.parquet',
                      output_dir, config)

    # --- Stage 5: Normalization + Splits ---
    if args.start_stage <= 5 <= args.end_stage:
        if df is not None:
            run_stage5(df, output_dir, config)
        else:
            run_stage5(output_dir / 'engineered_features.parquet',
                      output_dir, config)

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs (%.1f min)",
                total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)

    # Final output summary
    logger.info("\nOutput files:")
    for name in ['cleaned_features.parquet', 'engineered_features.parquet',
                 'normalized_features.parquet', 'eda_stats.parquet',
                 'eda_report.json', 'scalers.pkl', 'split_indices.pkl']:
        p = output_dir / name
        if p.exists():
            logger.info("  %s (%.1f MB)", p, p.stat().st_size / 1e6)
    for name in ['train.parquet', 'val.parquet', 'test.parquet']:
        p = output_dir / 'splits' / name
        if p.exists():
            logger.info("  splits/%s (%.1f MB)", name, p.stat().st_size / 1e6)


if __name__ == '__main__':
    main()
