"""Contracts of the data pipeline (Codex audit batch 1, items DATA-001 to DATA-016)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.batch_extract as batch_extract
from src.data.benchmark_logs import load_manifest, manifest_row
from src.data.benchmark_verify import DIMENSION_NAMES, verify_benchmark_log
from src.data.drishti_labeling import codes_to_labels, compute_drishti_codes
from src.data.feature_extraction import get_feature_names
from src.data.parse_darshan import _read_module_frames, parse_darshan_log
from src.data.preprocessing import (
    create_splits, engineer_one, load_preprocessing_config, stage2_clean, stage3_engineer,
    stage5_normalize)
from tests.pipeline_fixtures import raw_frame

CONFIG = {'cleaning': {'min_duration_seconds': 1, 'min_total_bytes': 1024, 'min_io_ops': 1,
                       'require_posix': True},
          'sentinel_handling': {'replace_negative_rank_with': 0},
          'normalization': {},
          'splits': {'method': 'temporal', 'test_fraction': 0.2, 'val_fraction': 0.2},
          'random_seed': 42}


def _raises(fn, exc, text=''):
    try:
        fn()
    except exc as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f'expected {exc.__name__}')


# --- DATA-001 / DATA-002: batch extraction resume and accounting ---------

def _fake_extract(path):
    if 'bad' in path:
        raise ValueError(f'synthetic failure for {path}')
    return {'_source_path': path, 'nprocs': 1, 'POSIX_WRITES': 1.0}


def test_batch_extract_resumes_by_identity_and_accounts_for_every_path():
    real = batch_extract.extract_single_log
    batch_extract.extract_single_log = _fake_extract
    try:
        with tempfile.TemporaryDirectory() as tmp:
            files = [f'/logs/{n}.darshan' for n in ('a', 'b', 'c', 'bad_d')]
            lst = Path(tmp) / 'list.txt'
            lst.write_text('\n'.join(files[:2]) + '\n')
            out = Path(tmp) / 'chunk.parquet'
            first = batch_extract.batch_extract(file_list=lst, output_path=out, max_workers=2,
                                                chunk_size=1, shuffle=False)
            assert first['n_rows'] == 2 and out.exists()
            # rerun with two more paths: the first two are skipped, one fails
            lst.write_text('\n'.join(files) + '\n')
            second = batch_extract.batch_extract(file_list=lst, output_path=out, max_workers=2,
                                                 chunk_size=1, shuffle=False)
            assert second['n_skipped'] == 2 and second['n_success'] == 1 and second['n_failed'] == 1
            published = pd.read_parquet(out)['_source_path'].tolist()
            assert sorted(published) == sorted(files[:3])
            errors = pd.read_csv(Path(tmp) / 'chunk_errors.csv')
            assert errors['file_path'].tolist() == [files[3]] and 'synthetic failure' in errors['error'][0]
            # a list with a path that is neither in a part nor in this run's failures is refused
            lst.write_text('/logs/a.darshan\n')
            _raises(lambda: batch_extract.batch_extract(file_list=lst, output_path=out, max_workers=1,
                                                        shuffle=False),
                    batch_extract.ExtractionError, 'not in the input')
    finally:
        batch_extract.extract_single_log = real


def test_batch_extract_refuses_empty_input_and_total_failure():
    real = batch_extract.extract_single_log
    batch_extract.extract_single_log = _fake_extract
    try:
        with tempfile.TemporaryDirectory() as tmp:
            lst = Path(tmp) / 'list.txt'
            lst.write_text('')
            _raises(lambda: batch_extract.batch_extract(file_list=lst, output_path=Path(tmp) / 'o.parquet'),
                    batch_extract.ExtractionError, 'no .darshan files')
            lst.write_text('/logs/bad_1.darshan\n')
            _raises(lambda: batch_extract.batch_extract(file_list=lst, output_path=Path(tmp) / 'o.parquet',
                                                        max_workers=1, shuffle=False),
                    batch_extract.ExtractionError, 'no successful extraction')
            assert not (Path(tmp) / 'o.parquet').exists()
    finally:
        batch_extract.extract_single_log = real


# --- DATA-003: a module that cannot be read fails the sample --------------

class _BrokenReport:
    modules = {'POSIX': {}}
    records = {}

    def mod_read_all_records(self, mod):
        raise RuntimeError('synthetic module read failure')


def test_unreadable_module_is_an_error_not_a_zero_sample():
    _raises(lambda: _read_module_frames(_BrokenReport(), 'POSIX', 'x.darshan'), ValueError,
            'cannot read module POSIX')
    assert parse_darshan_log('/nonexistent.darshan') is None
    _raises(lambda: parse_darshan_log('/nonexistent.darshan', strict=True), Exception)


# --- DATA-008: verification is a gate ------------------------------------

def _features(**over):
    # a healthy 4-rank job: 2000 sequential 4 MiB POSIX writes, two files, no MPI-IO
    f = engineer_one({'job': {'nprocs': 4, 'runtime': 30.0}, 'counters': {}, 'modules': ['POSIX'],
                      'shared_file_flags': {'POSIX': False}})
    f.update(nprocs=4, runtime_seconds=30.0, POSIX_READS=0.0, POSIX_WRITES=2000.0,
             POSIX_BYTES_WRITTEN=2000 * 4 * 1048576.0, POSIX_SIZE_WRITE_4M_10M=2000.0,
             POSIX_SEQ_WRITES=2000.0, metadata_time_ratio=0.01, rank_byte_range_ratio=0.0,
             SHARED_BYTE_IMBALANCE=0.0, num_files=2, is_shared_file=0, POSIX_FSYNCS=0.0)
    f.update(over)
    return f


CONTEXT = {'log_paths': [], 'offsets': {}, 'data_files': 2}


def test_verification_rejects_empty_labels_and_checks_every_healthy_condition():
    _raises(lambda: verify_benchmark_log(_features(), {d: 0 for d in DIMENSION_NAMES}, CONTEXT),
            ValueError, 'no dimension')
    healthy = {d: int(d == 'healthy') for d in DIMENSION_NAMES}
    passed, report = verify_benchmark_log(_features(), healthy, CONTEXT)
    assert passed and report['total_checks'] == 8
    passed, report = verify_benchmark_log(_features(POSIX_FSYNCS=2000.0), healthy, CONTEXT)
    assert not passed and report['checks']['healthy/no_throughput_utilization']['status'] == 'fail'
    # one fsync per rank at close (IOR -e) is not a sync-per-write construction
    assert verify_benchmark_log(_features(POSIX_FSYNCS=4.0), healthy, CONTEXT)[0]
    # a sub-second run is reported against the cleaning rule but not failed
    passed, report = verify_benchmark_log(_features(runtime_seconds=0.4), healthy, CONTEXT)
    assert passed and not report['cleaning_rule'] and 'runtime' in report['cleaning_reason']


def test_verification_has_a_rule_for_every_bottleneck_dimension():
    cases = {
        # Drishti P06: over 1000 small writes that are over 10 percent of the writes
        'access_granularity': dict(POSIX_SIZE_WRITE_4M_10M=0.0, POSIX_SIZE_WRITE_1K_10K=2000.0),
        'metadata_intensity': dict(metadata_time_ratio=0.5),
        'parallelism_efficiency': dict(rank_byte_range_ratio=0.9),
        'access_pattern': dict(POSIX_SEQ_WRITES=100.0),
        'interface_choice': dict(is_shared_file=1),
        'file_strategy': dict(),
        'throughput_utilization': dict(POSIX_FSYNCS=2000.0),
    }
    for dim, over in cases.items():
        labels = {d: int(d == dim) for d in DIMENSION_NAMES}
        context = dict(CONTEXT, data_files=4) if dim == 'file_strategy' else CONTEXT
        assert verify_benchmark_log(_features(**over), labels, context)[0], dim
        assert not verify_benchmark_log(_features(), labels, CONTEXT)[0], dim
    # a read-only job is not a metadata-only job
    labels = {d: int(d == 'metadata_intensity') for d in DIMENSION_NAMES}
    assert not verify_benchmark_log(_features(POSIX_BYTES_WRITTEN=0.0, POSIX_BYTES_READ=1e9, POSIX_READS=2000.0,
                                              POSIX_WRITES=0.0, POSIX_SEQ_READS=2000.0), labels, CONTEXT)[0]


def test_rules_follow_the_application_layer_for_mpiio_jobs():
    # collective buffering: 4 MiB MPI-IO collective writes become POSIX chunks under 1 MiB
    # on two aggregator ranks; judged at the MPI-IO layer the job is large and collective
    mpiio = dict(is_shared_file=1, MPIIO_COLL_WRITES=2000.0, MPIIO_SIZE_WRITE_AGG_4M_10M=2000.0,
                 POSIX_WRITES=8000.0, POSIX_SEQ_WRITES=8000.0, POSIX_SIZE_WRITE_4M_10M=0.0,
                 POSIX_SIZE_WRITE_100K_1M=8000.0, SHARED_BYTE_IMBALANCE=0.0)
    healthy = {d: int(d == 'healthy') for d in DIMENSION_NAMES}
    assert verify_benchmark_log(_features(**mpiio), healthy, CONTEXT)[0]
    # independent MPI-IO at scale without collectives is Drishti's M03
    indep = dict(mpiio, MPIIO_COLL_WRITES=0.0, MPIIO_INDEP_WRITES=2000.0)
    labels = {d: int(d == 'interface_choice') for d in DIMENSION_NAMES}
    assert verify_benchmark_log(_features(**indep), labels, CONTEXT)[0]
    # under 1000 MPI-IO operations Drishti does not fire, so a small independent job stays healthy
    few = dict(indep, MPIIO_INDEP_WRITES=145.0, MPIIO_SIZE_WRITE_AGG_4M_10M=145.0)
    assert verify_benchmark_log(_features(**few), healthy, CONTEXT)[0]


# --- DATA-007: manifest lookups are exact ---------------------------------

def test_manifest_requires_exactly_one_row_per_sample():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'manifest.csv'
        cols = ['benchmark', 'job_id', 'log_file', 'scenario', 'source', 'note'] + DIMENSION_NAMES
        rows = [['ior', '1', 'a.darshan', 's', 'slurm_out', ''] + [1, 0, 0, 0, 0, 0, 0, 0],
                ['ior', '2', 'b.darshan', '', 'none', 'no label'] + [0] * 8,
                ['custom', '3', '', 'c', 'slurm_out', ''] + [0, 0, 1, 0, 0, 0, 0, 0]]
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        m = load_manifest(path)
        assert manifest_row(m, 'ior', '1', ['/x/a.darshan'])['access_granularity'] == 1
        assert manifest_row(m, 'ior', '2', ['/x/b.darshan']) is None
        assert manifest_row(m, 'custom', '3', ['/x/p0.darshan', '/x/p1.darshan'])['parallelism_efficiency'] == 1
        _raises(lambda: manifest_row(m, 'ior', '9', ['/x/z.darshan']), KeyError, 'no manifest row')
        rows.append(rows[0])
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        _raises(lambda: load_manifest(path), ValueError, 'duplicate')


# --- DATA-010 / DATA-011 / DATA-012: schema contracts ----------------------

def test_feature_name_api_matches_real_extraction():
    engineered = engineer_one({'job': {'nprocs': 2, 'runtime': 5.0}, 'counters': {},
                               'modules': ['POSIX'], 'shared_file_flags': {}})
    produced = [k for k in engineered if not k.startswith('_')]
    assert sorted(produced) == sorted(get_feature_names())
    assert len(set(get_feature_names())) == len(get_feature_names()) == len(produced)


def test_preprocessing_refuses_old_or_incomplete_schema():
    df = raw_frame(3)
    old = df.copy()
    old['_schema_version'] = 1
    _raises(lambda: stage3_engineer(old), ValueError, 'schema version')
    _raises(lambda: stage2_clean(old, CONFIG), ValueError, 'schema version')
    partial = df.drop(columns=['RANK_BYTES_MAX'])
    _raises(lambda: stage3_engineer(partial), ValueError, 'RANK_BYTES_MAX')
    assert len(stage3_engineer(df)) == 3


def test_missing_scaler_and_missing_config_are_errors():
    df = stage3_engineer(raw_frame(3))
    _raises(lambda: stage5_normalize(df, CONFIG, fit=False, scalers={}), ValueError, 'no fitted scaler')
    _raises(lambda: load_preprocessing_config('/nonexistent/preprocessing.yaml'), FileNotFoundError)


# --- DATA-013: splits are positions --------------------------------------

def test_temporal_split_returns_positions_for_any_index():
    df = raw_frame(10).set_index(pd.Index([2, 3, 5, 6, 8, 9, 11, 12, 14, 15]))
    splits = create_splits(df, CONFIG)
    parts = np.concatenate([splits['train_idx'], splits['val_idx'], splits['test_idx']])
    assert sorted(parts) == list(range(10))
    assert list(splits['test_idx']) == [8, 9] and list(splits['val_idx']) == [6, 7]
    _raises(lambda: create_splits(raw_frame(3), CONFIG), ValueError, 'empty partition')
    cleaned, _ = stage2_clean(raw_frame(4, start=5), CONFIG)
    assert list(cleaned.index) == [0, 1, 2, 3]


# --- DATA-016: label conversion keeps the caller's index ------------------

def test_codes_to_labels_keeps_index_and_stays_binary():
    df = stage3_engineer(raw_frame(2)).set_index(pd.Index([10, 20]))
    labels = codes_to_labels(compute_drishti_codes(df))
    assert list(labels.index) == [10, 20]
    assert labels.isin([0, 1]).all().all() and labels['healthy'].tolist() == [1, 1]
