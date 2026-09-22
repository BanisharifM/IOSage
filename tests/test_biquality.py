"""Contracts of the biquality training path (Codex audit batch 2)."""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES
from src.models import biquality
from tests.pipeline_fixtures import raw_frame

RNG = np.random.RandomState(0)


def _config(tmp, **over):
    cfg = {
        'paths': {k: str(Path(tmp) / f'{k}.parquet') for k in
                  ('production_features', 'production_labels', 'benchmark_features', 'benchmark_labels')},
        'exclude_features': [], 'benchmark_split': {'test_ratio': 0.5, 'seed': 1},
        'biquality': {'clean_weight': 10.0, 'seeds': [1]}, 'decision_threshold': 0.5,
        'imbalance': {'max_weight': 100.0}, 'early_stopping': {'rounds': 5},
        'models': {'xgboost': {'params': {'n_estimators': 20, 'max_depth': 2, 'learning_rate': 0.3,
                                          'objective': 'binary:logistic', 'tree_method': 'hist',
                                          'eval_metric': 'logloss'}},
                   'lightgbm': {'params': {}}, 'random_forest': {'params': {'n_estimators': 10}}},
        'evaluation': {'bootstrap': {'n_resamples': 50, 'confidence_level': 0.95, 'seed': 0}},
        'runs_dir': tmp,
    }
    cfg['paths']['production_splits'] = str(Path(tmp) / 'split_indices.pkl')
    cfg['paths']['runs_dir'] = tmp
    cfg.update(over)
    cfg['_path'] = str(Path(tmp) / 'config.yaml')
    with open(cfg['_path'], 'w') as fh:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith('_')}, fh)
    return cfg


def _production(tmp, n=60):
    df = raw_frame(n)
    df['_source_path'] = [f'/logs/{i}.darshan' for i in range(n)]
    df['POSIX_BYTES_WRITTEN'] = RNG.uniform(1e3, 1e9, n)
    df['POSIX_WRITES'] = RNG.randint(1, 5000, n).astype(float)
    labels = pd.DataFrame({d: RNG.randint(0, 2, n) for d in BOTTLENECK_DIMENSIONS})
    labels['healthy'] = (labels[BOTTLENECK_DIMENSIONS].sum(axis=1) == 0).astype(int)
    labels['_source_path'] = df['_source_path']
    labels['_jobid'] = 7  # one job id for every row: alignment must not use it
    return df, labels


def _benchmark(tmp, n=40):
    df = raw_frame(n)
    df['_source_path'] = [f'/bench/log{i}.darshan' for i in range(n)]
    df['_benchmark'] = 'ior'
    df['_ground_truth_job_id'] = [str(100 + i // 2) for i in range(n)]  # two logs per job
    df['_scenario'] = 's'
    df['POSIX_BYTES_WRITTEN'] = RNG.uniform(1e3, 1e9, n)
    labels = pd.DataFrame({d: 0 for d in DIMENSION_NAMES}, index=range(n))
    labels['access_granularity'] = (np.arange(n) % 3 == 0).astype(int)
    labels['access_pattern'] = (np.arange(n) % 5 == 0).astype(int)
    labels['healthy'] = (labels[BOTTLENECK_DIMENSIONS].sum(axis=1) == 0).astype(int)
    labels['job_id'] = df['_ground_truth_job_id']
    labels['benchmark'] = 'ior'
    return df, labels


def _write(cfg, prod, prod_labels, bench, bench_labels, splits):
    prod.to_parquet(cfg['paths']['production_features'], index=False)
    prod_labels.to_parquet(cfg['paths']['production_labels'], index=False)
    bench.to_parquet(cfg['paths']['benchmark_features'], index=False)
    bench_labels.to_parquet(cfg['paths']['benchmark_labels'], index=False)
    with open(cfg['paths']['production_splits'], 'wb') as fh:
        pickle.dump(splits, fh)


def _raises(fn, exc, text=''):
    try:
        fn()
    except exc as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f'expected {exc.__name__}')


def test_grouped_split_keeps_jobs_together():
    y = RNG.randint(0, 2, (40, 7)).astype(float)
    groups = np.array([f'ior/{i // 2}' for i in range(40)])
    dev, test = biquality.grouped_benchmark_split(y, groups, 0.5, 3)
    assert not set(groups[dev]) & set(groups[test])
    assert sorted(np.concatenate([dev, test])) == list(range(40))


def test_healthy_is_derived_from_the_seven_decisions():
    class _P:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])

    bundle = {'feature_names': ['a', 'b'], 'decision_threshold': 0.5,
              'models': {d: _P(0.1) for d in BOTTLENECK_DIMENSIONS}}
    _, dec = biquality.predict(bundle, np.zeros((3, 2)))
    assert dec[:, 7].tolist() == [1, 1, 1] and dec[:, :7].sum() == 0
    bundle['models']['access_pattern'] = _P(0.9)
    _, dec = biquality.predict(bundle, np.zeros((3, 2)))
    assert dec[:, 7].tolist() == [0, 0, 0] and dec[:, 3].tolist() == [1, 1, 1]
    _raises(lambda: biquality.predict(bundle, np.zeros((3, 5))), ValueError, 'expected 2 features')


def test_production_alignment_uses_source_path_and_checks_time_order():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp)
        prod, labels = _production(tmp)
        bench, bench_labels = _benchmark(tmp)
        order = np.arange(60)
        splits = {'train_idx': order[:40], 'val_idx': order[40:50], 'test_idx': order[50:]}
        _write(cfg, prod, labels.iloc[::-1].reset_index(drop=True), bench, bench_labels, splits)
        data = biquality.load_production(cfg)
        # labels were written in reverse order: alignment by _source_path restores them
        expected = labels.set_index('_source_path').loc[prod['_source_path']][BOTTLENECK_DIMENSIONS].to_numpy()
        assert np.array_equal(data.y, expected)
        # a split that is not in time order is refused
        bad = {'train_idx': order[10:50], 'val_idx': order[:10], 'test_idx': order[50:]}
        _write(cfg, prod, labels, bench, bench_labels, bad)
        _raises(lambda: biquality.load_production(cfg), ValueError, 'time order')
        # a split that misses rows is refused
        _write(cfg, prod, labels, bench, bench_labels, {'train_idx': order[:40], 'val_idx': order[40:50], 'test_idx': order[50:55]})
        _raises(lambda: biquality.load_production(cfg), ValueError, 'cover every row')


def test_benchmark_loader_refuses_missing_contract_columns():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp)
        prod, labels = _production(tmp)
        bench, bench_labels = _benchmark(tmp)
        order = np.arange(60)
        _write(cfg, prod, labels, bench.drop(columns=['RANK_BYTES_MAX']), bench_labels,
               {'train_idx': order[:40], 'val_idx': order[40:50], 'test_idx': order[50:]})
        names = biquality.feature_columns(prod, cfg)
        _raises(lambda: biquality.load_benchmark(cfg, names), ValueError, 'RANK_BYTES_MAX')


def test_train_run_writes_bundle_manifest_and_refuses_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp)
        prod, labels = _production(tmp)
        bench, bench_labels = _benchmark(tmp)
        order = np.arange(60)
        _write(cfg, prod, labels, bench, bench_labels,
               {'train_idx': order[:40], 'val_idx': order[40:50], 'test_idx': order[50:]})
        run_dir = Path(tmp) / 'run1'
        manifest = biquality.train_run(cfg, 'xgboost', [1], 10.0, run_dir)
        bundle = pickle.load(open(run_dir / 'xgboost_w10_seed1.pkl', 'rb'))
        assert bundle['bundle_format'] == biquality.BUNDLE_FORMAT
        assert bundle['feature_names'] == biquality.feature_columns(prod, cfg)
        assert set(bundle['models']) == set(BOTTLENECK_DIMENSIONS)
        assert all(0 <= v < 20 for v in bundle['best_iteration'].values())   # early stopping recorded
        assert set(manifest['per_seed']['1']) == {'dev', 'test', 'best_iteration', 'fit_seconds'}
        assert manifest['sizes']['bench_dev'] + manifest['sizes']['bench_test'] == 40
        assert (run_dir / 'manifest.json').exists() and (run_dir / 'splits.npz').exists()
        _raises(lambda: biquality.train_run(cfg, 'xgboost', [1], 10.0, run_dir), FileExistsError)
        _raises(lambda: biquality.fit_models(prod[[]].to_numpy(), None, None, None, None, 'mlp', cfg, 1),
                ValueError, 'not supported')


def test_ablation_options_are_applied_and_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp)
        prod, labels = _production(tmp)
        bench, bench_labels = _benchmark(tmp)
        bench.loc[:9, '_benchmark'] = 'custom'
        bench.loc[:9, '_ground_truth_job_id'] = ['9' + str(i // 2) for i in range(10)]
        bench_labels.loc[:9, 'benchmark'] = 'custom'
        bench_labels.loc[:9, 'job_id'] = bench.loc[:9, '_ground_truth_job_id']
        order = np.arange(60)
        _write(cfg, prod, labels, bench, bench_labels,
               {'train_idx': order[:40], 'val_idx': order[40:50], 'test_idx': order[50:]})
        raw = biquality.train_run(cfg, 'xgboost', [1], 10.0, Path(tmp) / 'raw', feature_set='raw')
        assert raw['feature_set'] == 'raw'
        assert not any(n.endswith('_all') or n.startswith('avg_') for n in raw['feature_names'])
        gt_only = biquality.train_run(cfg, 'xgboost', [1], 10.0, Path(tmp) / 'gt', use_production=False)
        assert gt_only['sizes']['prod_train'] == 0 and gt_only['sizes']['bench_dev'] > 0
        lobo = biquality.train_run(cfg, 'xgboost', [1], 10.0, Path(tmp) / 'lobo', hold_out_benchmark='custom')
        splits = np.load(Path(tmp) / 'lobo' / 'splits.npz', allow_pickle=True)
        assert not any(g.startswith('custom/') for g in splits['bench_groups'][splits['bench_dev']])
        assert 'test_excluded_benchmark' in lobo['per_seed']['1'] and lobo['sizes']['bench_test_excluded'] > 0
        _raises(lambda: biquality.train_run(cfg, 'xgboost', [1], 10.0, Path(tmp) / 'x', hold_out_benchmark='none'),
                ValueError, 'no benchmark rows')
