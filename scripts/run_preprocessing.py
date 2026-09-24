"""
Preprocessing Pipeline Runner
==============================
Orchestrates Stages 2-5 of the preprocessing pipeline on raw_features.parquet.

Stages:
  2. Cleaning        -> data/processed/resubmission/production/cleaned_features.parquet
  3. Engineering      -> data/processed/resubmission/production/features.parquet
  4. EDA / Statistics -> data/processed/resubmission/production/eda_stats.parquet + eda_report.json
  5. Normalization    -> data/processed/resubmission/production/splits/*.parquet
                      + data/processed/resubmission/production/scalers.pkl

Usage::

    python scripts/run_preprocessing.py \
        --input data/processed/resubmission/production/raw_features.parquet \
        --output-dir data/processed/resubmission/production \
        --config configs/preprocessing.yaml

    # Resume from a specific stage (read its prerequisite from output-dir)
    python scripts/run_preprocessing.py \
        --output-dir data/processed/resubmission/production \
        --start-stage 3

    # Quick test on sample
    python scripts/run_preprocessing.py \
        --input data/processed/resubmission/production/raw_features.parquet \
        --output-dir data/processed/resubmission/production \
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
from src.data.feature_extraction import FEATURE_SCHEMA_VERSION
from src.utils.artifacts import sha256_file, write_atomic

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


def write_parquet(df, path, index=False):
    """Atomic parquet write; an existing file is an error."""
    return write_atomic(path, lambda temporary: df.to_parquet(temporary, index=index))


def write_json(payload, path):
    def writer(temporary):
        with open(temporary, 'w') as fh:
            json.dump(payload, fh, indent=2)
            fh.write('\n')
    return write_atomic(path, writer)


def write_pickle(payload, path):
    def writer(temporary):
        with open(temporary, 'wb') as fh:
            pickle.dump(payload, fh)
    return write_atomic(path, writer)


def sample_rows(df, sample, seed):
    """A seeded row sample of ``df`` (positions reset) or ``df`` unchanged."""
    if sample is None or len(df) <= sample:
        return df
    positions = np.random.RandomState(seed).choice(len(df), sample, replace=False)
    logger.info("Sampled %d of %d rows for testing", sample, len(df))
    return df.iloc[np.sort(positions)].reset_index(drop=True)


def validate_dataframe(df, stage_name, expected_min_rows):
    """Check a stage's output for NaN, infinity and a minimum row count.

    Raises ``ValueError`` on any issue, so a stage never publishes an invalid
    frame and the process ends with a nonzero status.
    """
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
        raise ValueError(f"[{stage_name}] validation failed: " + "; ".join(issues))
    logger.info("[%s] VALIDATION PASSED: %d rows, %d columns, "
                "%d all-zero features, no NaN/Inf",
                stage_name, len(df), len(df.columns), n_all_zero)


def run_stage2(input_path, output_dir, config, min_rows):
    """Stage 2: Cleaning."""
    logger.info("=" * 60)
    logger.info("STAGE 2: CLEANING")
    logger.info("=" * 60)

    t0 = time.time()
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows x %d columns from %s",
                len(df), len(df.columns), input_path)

    df_clean, report = stage2_clean(df, config)
    validate_dataframe(df_clean, "Stage 2", expected_min_rows=min_rows)

    # Save
    out_path = output_dir / 'cleaned_features.parquet'
    write_parquet(df_clean, out_path)

    elapsed = time.time() - t0
    logger.info("Stage 2 complete in %.1fs: %d -> %d rows (removed %d, %.1f%%)",
                elapsed, report['initial_rows'], report['final_rows'],
                report['rows_removed'], report['removal_pct'])
    logger.info("Saved: %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)

    # Log detailed report
    for key, val in report.items():
        logger.info("  %s: %s", key, val)

    return df_clean, report


def run_stage3(df, output_dir, config, min_rows):
    """Stage 3: Feature Engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    t0 = time.time()
    n_before = len(df.columns)
    df_eng = stage3_engineer(df, config=config)
    n_after = len(df_eng.columns)
    validate_dataframe(df_eng, "Stage 3", expected_min_rows=min_rows)

    # Save
    out_path = output_dir / 'features.parquet'
    write_parquet(df_eng, out_path)

    elapsed = time.time() - t0
    logger.info("Stage 3 complete in %.1fs: %d -> %d columns (+%d derived)",
                elapsed, n_before, n_after, n_after - n_before)
    logger.info("Saved: %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)

    return df_eng


def run_stage4(df, output_dir, config):
    """Stage 4: Statistical Analysis (EDA)."""
    logger.info("=" * 60)
    logger.info("STAGE 4: STATISTICAL ANALYSIS (EDA)")
    logger.info("=" * 60)

    t0 = time.time()

    # Per-feature statistics
    logger.info("Computing per-feature statistics...")
    stats = compute_statistics(df)
    stats_path = output_dir / 'eda_stats.parquet'
    write_parquet(stats, stats_path, index=True)
    logger.info("Saved feature statistics: %s (%d features)", stats_path,
                len(stats))

    # Correlation matrix (Spearman handles non-linear monotonic relationships)
    logger.info("Computing Spearman correlation matrix...")
    feature_sel = config['feature_selection']
    corr_threshold = feature_sel['correlation_threshold']
    min_nonzero = feature_sel['min_nonzero_fraction']

    corr_matrix = compute_correlation_matrix(df, method='spearman')
    corr_path = output_dir / 'eda_correlation.parquet'
    write_parquet(corr_matrix, corr_path, index=True)

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
    write_json(eda_report, report_path)

    elapsed = time.time() - t0
    logger.info("Stage 4 complete in %.1fs", elapsed)
    logger.info("Saved: %s, %s, %s",
                stats_path.name, corr_path.name, report_path.name)

    return stats, eda_report


def run_stage5(df, output_dir, config, min_rows):
    """Stage 5: Normalization + Splits (split arrays are row positions)."""
    logger.info("=" * 60)
    logger.info("STAGE 5: NORMALIZATION + SPLITS")
    logger.info("=" * 60)

    t0 = time.time()

    # Create splits before normalization so scalers see training rows only.
    logger.info("Creating train/val/test splits...")
    splits = create_splits(df, config)
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(train_idx), len(val_idx), len(test_idx))

    # Drop excluded features (constant + manually listed in config)
    # Use train set as reference for detecting constant features
    df_train_ref = df.iloc[train_idx]
    n_before = len([c for c in df.columns if not c.startswith('_')])
    df, dropped_features = drop_excluded_features(
        df, config, train_df=df_train_ref)
    n_after = len([c for c in df.columns if not c.startswith('_')])
    logger.info("Feature exclusion: %d -> %d features (dropped %d)",
                n_before, n_after, len(dropped_features))

    # Save dropped feature list for reference
    dropped_path = output_dir / 'dropped_features.json'
    write_json({
        'dropped': dropped_features,
        'count': len(dropped_features),
        'remaining': n_after,
    }, dropped_path)

    # Normalize TRAINING set (fit scalers)
    logger.info("Normalizing training set (fitting scalers)...")
    df_train = df.iloc[train_idx].copy()
    df_train_norm, scalers = stage5_normalize(df_train, config, fit=True)

    # Normalize VAL and TEST with pre-fitted scalers
    logger.info("Normalizing validation set (transform only)...")
    df_val = df.iloc[val_idx].copy()
    df_val_norm, _ = stage5_normalize(df_val, config, fit=False, scalers=scalers)

    logger.info("Normalizing test set (transform only)...")
    df_test = df.iloc[test_idx].copy()
    df_test_norm, _ = stage5_normalize(df_test, config, fit=False, scalers=scalers)

    # Validate before anything is written
    split_config = config['splits']
    partition_minimums = {
        'train': max(1, int(min_rows * (1 - split_config['val_fraction']
                                        - split_config['test_fraction']))),
        'val': max(1, int(min_rows * split_config['val_fraction'])),
        'test': max(1, int(min_rows * split_config['test_fraction'])),
    }
    for name, df_norm in [('train', df_train_norm), ('val', df_val_norm),
                          ('test', df_test_norm)]:
        validate_dataframe(df_norm, f"Stage 5 ({name})",
                          expected_min_rows=partition_minimums[name])

    # Save splits
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(df_train_norm, splits_dir / 'train.parquet')
    write_parquet(df_val_norm, splits_dir / 'val.parquet')
    write_parquet(df_test_norm, splits_dir / 'test.parquet')

    # Save scalers and split indices
    write_pickle(scalers, output_dir / 'scalers.pkl')
    write_pickle(splits, output_dir / 'split_indices.pkl')

    # Also save the full normalized dataset (train scalers applied to all)
    logger.info("Creating full normalized dataset...")
    df_full_norm, _ = stage5_normalize(
        df.copy(), config, fit=False, scalers=scalers)
    write_parquet(df_full_norm, output_dir / 'normalized_features.parquet')

    elapsed = time.time() - t0
    logger.info("Stage 5 complete in %.1fs", elapsed)

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
                        default='data/processed/resubmission/production/raw_features.parquet',
                        help='Input parquet from Stage 1')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed/resubmission/production',
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
    parser.add_argument('--min-rows', type=int, default=100000,
                        help='Rows every stage output must keep (validation)')
    args = parser.parse_args()

    if not 2 <= args.start_stage <= args.end_stage <= 5:
        parser.error("stages must satisfy 2 <= start-stage <= end-stage <= 5")
    input_path = Path(args.input).resolve()
    config_path = Path(args.config).resolve()
    if args.start_stage == 2 and not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if not config_path.is_file():
        parser.error(f"configuration file not found: {config_path}")

    # Setup
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_input_path = {
        2: input_path,
        3: output_dir / 'cleaned_features.parquet',
        4: output_dir / 'features.parquet',
        5: output_dir / 'features.parquet',
    }[args.start_stage]
    if not stage_input_path.is_file():
        parser.error(f"stage {args.start_stage} input file not found: {stage_input_path}")
    effective_min_rows = min(args.min_rows, args.sample) if args.sample else args.min_rows
    stage_outputs = {
        2: [output_dir / 'cleaned_features.parquet'],
        3: [output_dir / 'features.parquet'],
        4: [output_dir / 'eda_stats.parquet', output_dir / 'eda_correlation.parquet',
            output_dir / 'eda_report.json'],
        5: [output_dir / 'dropped_features.json', output_dir / 'normalized_features.parquet',
            output_dir / 'scalers.pkl', output_dir / 'split_indices.pkl',
            output_dir / 'splits' / 'train.parquet', output_dir / 'splits' / 'val.parquet',
            output_dir / 'splits' / 'test.parquet'],
    }
    manifest_name = ('preprocessing_manifest.json' if (args.start_stage, args.end_stage) == (2, 5)
                     else f'preprocessing_stage_{args.start_stage}_{args.end_stage}_manifest.json')
    manifest_path = output_dir / manifest_name
    selected_outputs = [path for stage in range(args.start_stage, args.end_stage + 1)
                        for path in stage_outputs[stage]]
    existing = [str(path) for path in selected_outputs + [manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace preprocessing artifacts: {existing}")
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("Input:      %s", stage_input_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Config:     %s", args.config)
    logger.info("Stages:     %d to %d", args.start_stage, args.end_stage)
    if args.sample:
        logger.info("Sample:     %d rows", args.sample)

    config = load_preprocessing_config(config_path)
    logger.info("Config loaded: %s", config_path)

    t_total = time.time()

    # Starting frame: stage 2 reads the raw parquet; a later start stage reads
    # its prerequisite. The optional sample is taken once, on that frame.
    if args.start_stage == 2:
        df, _ = run_stage2(input_path, output_dir, config, args.min_rows)
    else:
        df = pd.read_parquet(stage_input_path)
        logger.info("Loaded %d rows from %s", len(df), stage_input_path)
    df = sample_rows(df, args.sample, config['random_seed'])

    if args.start_stage <= 3 <= args.end_stage:
        df = run_stage3(df, output_dir, config, effective_min_rows)
    if args.start_stage <= 4 <= args.end_stage:
        run_stage4(df, output_dir, config)
    if args.start_stage <= 5 <= args.end_stage:
        run_stage5(df, output_dir, config, effective_min_rows)

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs (%.1f min)",
                total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)

    # Final output summary
    logger.info("\nOutput files:")
    for name in ['cleaned_features.parquet', 'features.parquet',
                 'normalized_features.parquet', 'eda_stats.parquet',
                 'eda_report.json', 'scalers.pkl', 'split_indices.pkl']:
        p = output_dir / name
        if p.exists():
            logger.info("  %s (%.1f MB)", p, p.stat().st_size / 1e6)
    for name in ['train.parquet', 'val.parquet', 'test.parquet']:
        p = output_dir / 'splits' / name
        if p.exists():
            logger.info("  splits/%s (%.1f MB)", name, p.stat().st_size / 1e6)

    artifacts = {}
    for path in selected_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required preprocessing output missing or empty: {path}")
        record = {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
        if path.suffix == '.parquet':
            frame = pd.read_parquet(path)
            record.update(rows=len(frame), columns=len(frame.columns))
        artifacts[path.relative_to(output_dir).as_posix()] = record
    manifest = {
        'schema_version': 1,
        'status': 'passed',
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'stages': [args.start_stage, args.end_stage],
        'sample': args.sample,
        'min_rows': args.min_rows,
        'effective_min_rows': effective_min_rows,
        'input': {'path': str(stage_input_path), 'sha256': sha256_file(stage_input_path)},
        'config': {'path': str(config_path), 'sha256': sha256_file(config_path)},
        'script': {'path': str(Path(__file__).resolve()),
                   'sha256': sha256_file(Path(__file__).resolve())},
        'artifacts': artifacts,
    }
    def write_manifest(path):
        with path.open('x') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')

    write_atomic(manifest_path, write_manifest)
    logger.info("Manifest: %s", manifest_path)


if __name__ == '__main__':
    main()
