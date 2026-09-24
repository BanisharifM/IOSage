"""Contracts of the data pipeline (Codex audit batch 1, items DATA-001 to DATA-016)."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.batch_extract as batch_extract
from scripts.build_label_manifest import apply_manifest_policy, label_string_to_dims
from src.data.benchmark_logs import (
    VALIDITY_COLUMNS, load_manifest, manifest_row, validate_verification_report,
)
from src.data.benchmark_verify import DIMENSION_NAMES, verify_benchmark_log
from src.data.drishti_labeling import (
    codes_to_labels, compute_drishti_codes, generate_heuristic_labels,
)
from src.data.feature_extraction import (
    ALL_RAW_COUNTERS, FEATURE_SCHEMA_VERSION, extract_raw_features, get_feature_names)
from src.data.label_rules import labels_from_features, rule_details, validity_from_features
from src.data.label_rules import (
    LABEL_DEFINITIONS, LABEL_DEFINITIONS_PATH, MANY_SMALL_FILES_COUNT,
    SMALL_FILE_MEAN_BYTES,
)
from src.data.parse_darshan import _partial_feature_modules, _read_module_frames, parse_darshan_log
from src.data.preprocessing import (
    create_splits, engineer_one, load_preprocessing_config, stage2_clean, stage3_engineer,
    stage5_normalize)
from tests.pipeline_fixtures import raw_frame

CONFIG = {'cleaning': {'min_duration_seconds': 1, 'min_total_bytes': 1024, 'min_io_ops': 1,
                       'require_posix': True},
          'sentinel_handling': {'replace_negative_rank_with': 0,
                                'replace_unavailable_counter_with': 0},
          'normalization': {
              'volume_counters': 'log1p_robust', 'count_counters': 'log1p_robust',
              'histogram_counters': 'log1p', 'top4_counters': 'log1p',
              'timing_counters': 'log1p_robust', 'timestamp_counters': 'none',
              'categorical_counters': 'none', 'rank_id_counters': 'none',
              'rank_stat_counters': 'log1p', 'rank_stat_bounded_counters': 'none',
              'conditional_size_counters': 'log1p', 'indicator_features': 'none',
              'ratio_features': 'none', 'ratio_unbounded_features': 'log1p',
              'derived_absolute': 'log1p', 'metadata_features': 'log1p'},
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
    return _raw_row(path)


def _fake_extract_all(path):
    return _raw_row(path)


def _raw_row(path):
    row = batch_extract.extract_raw_features({
        'job': {'nprocs': 1, 'runtime': 1.0}, 'counters': {},
        'modules': [], 'shared_file_flags': {},
    })
    row['_source_path'] = path
    return row


def _parsed_row(nprocs=4, runtime=30.0):
    raw = raw_frame(1).iloc[0]
    counters = {name: float(raw[name]) for name in ALL_RAW_COUNTERS}
    counters.update(num_files=2.0, num_data_files=2.0)
    return {
        'job': {'nprocs': nprocs, 'runtime': runtime},
        'counters': counters,
        'modules': ['POSIX'],
        'shared_file_flags': {'POSIX': False},
        'partial_modules': [],
    }


def test_batch_extract_resumes_by_identity_and_accounts_for_every_path():
    real = batch_extract.extract_single_log
    batch_extract.extract_single_log = _fake_extract
    try:
        with tempfile.TemporaryDirectory() as tmp:
            files = [f'/logs/{n}.darshan' for n in ('a', 'b', 'c', 'bad_d')]
            lst = Path(tmp) / 'list.txt'
            lst.write_text('\n'.join(files) + '\n')
            out = Path(tmp) / 'chunk.parquet'
            _raises(lambda: batch_extract.batch_extract(
                file_list=lst, output_path=out, max_workers=2, chunk_size=1, shuffle=False),
                batch_extract.ExtractionIncomplete, '1 requested paths failed')
            assert not out.exists()
            # A changed request cannot reuse the first attempt's parts.
            lst.write_text('\n'.join(files[:2]) + '\n')
            _raises(lambda: batch_extract.batch_extract(file_list=lst, output_path=out,
                                                        max_workers=1, shuffle=False),
                    batch_extract.ExtractionError, 'input set differs')
            # Retry the identical request after the failing path is repaired.
            lst.write_text('\n'.join(files) + '\n')
            batch_extract.extract_single_log = _fake_extract_all
            second = batch_extract.batch_extract(file_list=lst, output_path=out, max_workers=2,
                                                 chunk_size=1, shuffle=False)
            assert second['n_skipped'] == 3 and second['n_success'] == 1 and second['n_failed'] == 0
            published = pd.read_parquet(out)['_source_path'].tolist()
            assert sorted(published) == sorted(files)
            error_files = sorted(Path(tmp).glob('chunk_attempt_*_errors.csv'))
            assert len(error_files) == 2
            errors = pd.read_csv(error_files[0])
            assert errors['file_path'].tolist() == [files[3]] and 'synthetic failure' in errors['error'][0]
    finally:
        batch_extract.extract_single_log = real


def _fake_extract_or_die(path):
    if 'die' in path:
        os.kill(os.getpid(), signal.SIGBUS)   # what libdarshan-util does on a truncated log
    if 'slow' in path:
        time.sleep(30)
    return _raw_row(path)


def test_batch_extract_records_a_crashed_or_stuck_parse_and_finishes():
    real = batch_extract.extract_single_log
    batch_extract.extract_single_log = _fake_extract_or_die
    try:
        with tempfile.TemporaryDirectory() as tmp:
            files = [f'/logs/{n}.darshan' for n in ('a', 'die_b', 'slow_c', 'd')]
            lst = Path(tmp) / 'list.txt'
            lst.write_text('\n'.join(files) + '\n')
            out = Path(tmp) / 'chunk.parquet'
            started = time.monotonic()
            _raises(lambda: batch_extract.batch_extract(
                file_list=lst, output_path=out, max_workers=2, timeout_per_file=2,
                chunk_size=10, shuffle=False),
                batch_extract.ExtractionIncomplete, '2 requested paths failed')
            assert time.monotonic() - started < 25
            errors = pd.read_csv(sorted(Path(tmp).glob('chunk_attempt_*_errors.csv'))[0])
            recorded = dict(zip(errors['file_path'], errors['error']))
            assert 'SIGBUS' in recorded['/logs/die_b.darshan']
            assert recorded['/logs/slow_c.darshan'].startswith('timeout_after_2s')
            parts = sorted(Path(tmp).glob('chunk_part_*.parquet'))
            assert sorted(pd.read_parquet(parts[0])['_source_path']) == ['/logs/a.darshan', '/logs/d.darshan']
    finally:
        batch_extract.extract_single_log = real


def test_benchmark_samples_record_the_parse_cause_for_every_layout():
    from src.data.benchmark_logs import iter_benchmark_samples
    with tempfile.TemporaryDirectory() as tmp:
        for name in ('u_ior_id11-1_1-1-1-1_1.darshan', 'u_python_id22-1_1-1-1-1_1.darshan'):
            (Path(tmp) / name).write_bytes(b'not a darshan log')
        samples = list(iter_benchmark_samples('ior', tmp))
        assert [job for job, _, _, _ in samples] == ['11', '22']
        assert all(parsed is None for _, _, parsed, _ in samples)
        assert all(error.startswith('RuntimeError: ') for _, _, _, error in samples), samples
        merged = list(iter_benchmark_samples('custom', tmp))
        assert [job for job, _, _, _ in merged] == ['11', '22']
        assert all(error.startswith('ValueError: cannot open per-rank log') for _, _, _, error in merged)


def test_label_rules_import_without_the_native_darshan_library():
    code = 'import sys, src.data.label_rules, src.data.benchmark_verify; sys.exit(int("darshan" in sys.modules))'
    completed = subprocess.run([sys.executable, '-c', code], cwd=Path.cwd(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


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
                    batch_extract.ExtractionIncomplete, '1 requested paths failed')
            assert not (Path(tmp) / 'o.parquet').exists()
    finally:
        batch_extract.extract_single_log = real


# --- DATA-003: a module that cannot be read fails the sample --------------

class _BrokenReport:
    modules = {'POSIX': {}}
    records = {}

    def mod_read_all_records(self, mod):
        raise RuntimeError('synthetic module read failure')


class _PartialReport:
    modules = {'POSIX': {'partial_flag': True}}


def test_unreadable_module_is_an_error_not_a_zero_sample():
    _raises(lambda: _read_module_frames(_BrokenReport(), 'POSIX', 'x.darshan'), ValueError,
            'cannot read module POSIX')
    assert parse_darshan_log('/nonexistent.darshan') is None
    _raises(lambda: parse_darshan_log('/nonexistent.darshan', strict=True), Exception)
    assert _partial_feature_modules(_PartialReport()) == {'POSIX'}


def test_present_module_contract_is_strict_and_partial_state_is_explicit():
    parsed = _parsed_row()
    parsed['partial_modules'] = ['POSIX']
    features = extract_raw_features(parsed)
    assert features['partial_posix'] == 1 and features['partial_mpiio'] == 0
    engineered = stage3_engineer(pd.DataFrame([features]), config=CONFIG)
    validity = validity_from_features(engineered).iloc[0]
    assert validity['interface_choice'] == 1
    assert not validity[[d for d in DIMENSION_NAMES if d != 'interface_choice']].any()

    missing = _parsed_row()
    missing['counters'].pop('POSIX_FSYNCS')
    _raises(lambda: extract_raw_features(missing), ValueError, 'POSIX_FSYNCS')
    invalid_processes = _parsed_row()
    invalid_processes['job']['nprocs'] = 0
    _raises(lambda: extract_raw_features(invalid_processes), ValueError, 'nprocs')
    for bad_runtime in (-5.0, float('nan')):
        invalid_runtime = _parsed_row()
        invalid_runtime['job']['runtime'] = bad_runtime
        _raises(lambda: extract_raw_features(invalid_runtime), ValueError, 'runtime')
    assert extract_raw_features(_parsed_row(runtime=0.0))['runtime_seconds'] == 0.0


# --- DATA-008: verification is a gate ------------------------------------

def _features(**over):
    # a healthy 4-rank job: 2000 sequential 4 MiB POSIX writes, two files, no MPI-IO
    f = engineer_one(_parsed_row(), config=CONFIG)
    f.update(nprocs=4, runtime_seconds=30.0, POSIX_READS=0.0, POSIX_WRITES=2000.0,
             POSIX_BYTES_WRITTEN=2000 * 4 * 1048576.0, POSIX_SIZE_WRITE_4M_10M=2000.0,
             POSIX_SEQ_WRITES=2000.0,
             metadata_time_ratio=0.01, rank_byte_range_ratio=0.0,
             SHARED_BYTE_IMBALANCE=0.0, num_files=2, num_data_files=2,
             is_shared_file=0, POSIX_FSYNCS=0.0,
             io_bytes_all=2000 * 4 * 1048576.0, io_ops_all=2000.0,
             metadata_time_ratio_all=0.01)
    f.update(over)
    return f


def test_verification_rejects_empty_labels_and_checks_every_healthy_condition():
    _raises(lambda: verify_benchmark_log(
        _features(), {d: 0 for d in DIMENSION_NAMES}, CONFIG['cleaning']),
            ValueError, 'no valid problem target')
    healthy = {d: int(d == 'healthy') for d in DIMENSION_NAMES}
    passed, report = verify_benchmark_log(_features(), healthy, CONFIG['cleaning'])
    assert passed and report['total_checks'] == 9
    passed, report = verify_benchmark_log(
        _features(POSIX_FSYNCS=2000.0), healthy, CONFIG['cleaning'])
    assert not passed and report['checks']['throughput_utilization/rule']['status'] == 'fail'
    # one fsync per rank at close (IOR -e) is not a sync-per-write construction
    assert verify_benchmark_log(
        _features(POSIX_FSYNCS=4.0), healthy, CONFIG['cleaning'])[0]
    # a sub-second run is reported against the cleaning rule but not failed
    passed, report = verify_benchmark_log(
        _features(runtime_seconds=0.4), healthy, CONFIG['cleaning'])
    assert passed and not report['cleaning_rule'] and 'runtime' in report['cleaning_reason']


def test_verification_has_a_rule_for_every_bottleneck_dimension():
    cases = {
        # Drishti P06: over 1000 small writes that are over 10 percent of the writes
        'access_granularity': dict(POSIX_SIZE_WRITE_4M_10M=0.0, POSIX_SIZE_WRITE_1K_10K=2000.0),
        'metadata_intensity': dict(metadata_time_ratio_all=0.5),
        'parallelism_efficiency': dict(rank_byte_range_ratio=0.9),
        'access_pattern': dict(POSIX_SEQ_WRITES=100.0),
        'request_alignment': dict(POSIX_FILE_NOT_ALIGNED=1000.0),
        'interface_choice': dict(MPIIO_INDEP_WRITES=2000.0),
        'file_strategy': dict(num_data_files=1001, io_bytes_all=1001 * 4096.0),
        'throughput_utilization': dict(POSIX_FSYNCS=2000.0),
    }
    for dim, over in cases.items():
        labels = {d: int(d == dim) for d in DIMENSION_NAMES}
        assert verify_benchmark_log(
            _features(**over), labels, CONFIG['cleaning'])[0], dim
        assert not verify_benchmark_log(
            _features(), labels, CONFIG['cleaning'])[0], dim
    # Read-only access does not imply metadata intensity.
    labels = {d: int(d == 'metadata_intensity') for d in DIMENSION_NAMES}
    assert not verify_benchmark_log(
        _features(POSIX_BYTES_WRITTEN=0.0, POSIX_BYTES_READ=1e9,
                  POSIX_READS=2000.0, POSIX_WRITES=0.0, POSIX_SEQ_READS=2000.0,
                  io_bytes_all=1e9),
        labels, CONFIG['cleaning'])[0]


def test_rules_follow_the_application_layer_for_mpiio_jobs():
    # collective buffering: 4 MiB MPI-IO collective writes become POSIX chunks under 1 MiB
    # on two aggregator ranks; judged at the MPI-IO layer the job is large and collective
    mpiio = dict(is_shared_file=1, MPIIO_COLL_WRITES=2000.0, MPIIO_SIZE_WRITE_AGG_4M_10M=2000.0,
                 POSIX_WRITES=8000.0, POSIX_SEQ_WRITES=8000.0, POSIX_SIZE_WRITE_4M_10M=0.0,
                 POSIX_SIZE_WRITE_100K_1M=8000.0, SHARED_BYTE_IMBALANCE=0.0)
    healthy = {d: int(d == 'healthy') for d in DIMENSION_NAMES}
    assert verify_benchmark_log(_features(**mpiio), healthy, CONFIG['cleaning'])[0]
    # independent MPI-IO at scale without collectives is Drishti's M03
    indep = dict(mpiio, MPIIO_COLL_WRITES=0.0, MPIIO_INDEP_WRITES=2000.0)
    labels = {d: int(d == 'interface_choice') for d in DIMENSION_NAMES}
    assert verify_benchmark_log(_features(**indep), labels, CONFIG['cleaning'])[0]
    # under 1000 MPI-IO operations Drishti does not fire, so a small independent job stays healthy
    few = dict(indep, MPIIO_INDEP_WRITES=145.0, MPIIO_SIZE_WRITE_AGG_4M_10M=145.0)
    assert verify_benchmark_log(_features(**few), healthy, CONFIG['cleaning'])[0]

    # Layer selection is per direction: MPI-IO writes must not hide POSIX-only reads.
    mixed = dict(
        mpiio,
        POSIX_READS=2000.0,
        POSIX_SEQ_READS=2000.0,
        POSIX_SIZE_READ_1K_10K=2000.0,
    )
    labels = {d: int(d == 'access_granularity') for d in DIMENSION_NAMES}
    assert verify_benchmark_log(_features(**mixed), labels, CONFIG['cleaning'])[0]
    detail = rule_details(_features(**mixed))['access_granularity']
    assert 'read_posix_small=2000/2000' in detail
    assert 'write_mpiio_small=0/2000' in detail


def test_production_and_benchmark_labels_use_the_same_rules():
    rows = pd.DataFrame([
        _features(),
        _features(POSIX_SIZE_WRITE_4M_10M=0.0, POSIX_SIZE_WRITE_1K_10K=2000.0),
        _features(num_data_files=1001, io_bytes_all=1001 * 4096.0),
        _features(POSIX_FSYNCS=2000.0),
    ])
    expected = labels_from_features(rows)
    for position, feature_row in rows.iterrows():
        intended = expected.loc[position].to_dict()
        passed, report = verify_benchmark_log(
            feature_row.to_dict(), intended, CONFIG['cleaning'])
        assert passed, report


def test_label_definition_contract_is_complete_and_controls_file_scale():
    assert LABEL_DEFINITIONS_PATH.exists()
    assert list(LABEL_DEFINITIONS['dimensions']) == DIMENSION_NAMES
    assert MANY_SMALL_FILES_COUNT == 1000
    assert SMALL_FILE_MEAN_BYTES == 1048576
    required = {
        'definition', 'criterion', 'impact', 'canonical_fix',
        'when_not_to_apply', 'sources',
    }
    assert all(set(row) == required for row in LABEL_DEFINITIONS['dimensions'].values())
    assert not labels_from_features(
        _features(num_data_files=1000, io_bytes_all=1000 * 4096.0)
    ).iloc[0][
        'file_strategy'
    ]
    assert labels_from_features(
        _features(num_data_files=1001, io_bytes_all=1001 * 4096.0)
    ).iloc[0][
        'file_strategy'
    ]


def test_tracebench_mapping_keeps_distinct_expert_classes_separate():
    mapping = json.loads(Path(
        'data/external/tracebench/label_mapping.json'
    ).read_text())['tracebench_to_our_taxonomy']
    assert mapping['SML-R']['our_dimension'] == 'access_granularity'
    assert mapping['MSL-R']['our_dimension'] == 'request_alignment'
    assert mapping['NC-R']['our_dimension'] == 'interface_choice'
    for label in ('SHF', 'LLL-R', 'LLL-W', 'MPNM', 'SLIM', 'RDA-R'):
        assert mapping[label]['our_dimension'] is None


def test_manifest_policy_matches_registered_label_definitions():
    single_ost = label_string_to_dims(
        'access_granularity=1,interface_choice=1,throughput_utilization=1'
    )
    revised, validity, note = apply_manifest_policy(
        'h5bench', 'h5b_indep_small_single_ost_n32_r1', single_ost
    )
    assert revised is None and validity is None
    assert 'storage-layout confound' in note

    unsupported, validity, reason = apply_manifest_policy(
        'hacc_io', 'hacc_posix_shared_single_ost_n32_r1',
        label_string_to_dims('throughput_utilization=1'),
    )
    assert unsupported is None and validity is None
    assert 'does not isolate' in reason

    checkpoint, validity, reason = apply_manifest_policy(
        'dlio', 'dlio_ckpt_ms100000000_n4_rep1',
        label_string_to_dims('throughput_utilization=1'),
    )
    assert checkpoint is None and validity is None
    assert 'do not distinguish' in reason

    misaligned, validity, note = apply_manifest_policy(
        'ior', 'ior_misaligned_n16_r1',
        label_string_to_dims('access_granularity=1'),
    )
    assert misaligned['access_granularity'] == 1
    assert misaligned['request_alignment'] == 1
    assert validity['valid_access_granularity'] == 1
    assert validity['valid_request_alignment'] == 1
    assert validity['valid_metadata_intensity'] == 0
    assert 'request_alignment' in note

    incomplete, validity, reason = apply_manifest_policy(
        'mdtest', 'mdtest_meta_unique_n5000_r4_rep1',
        label_string_to_dims('metadata_intensity=1'),
    )
    assert incomplete is None and validity is None
    assert 'partial POSIX' in reason

    configured, validity, _ = apply_manifest_policy(
        'mdtest', 'mdtest_meta_unique_configured_n5000_r4_rep1',
        label_string_to_dims('metadata_intensity=1'),
    )
    assert configured['metadata_intensity'] == 1
    assert validity['valid_metadata_intensity'] == 1


def test_verification_rejects_incomplete_module_records():
    healthy = {d: int(d == 'healthy') for d in DIMENSION_NAMES}
    passed, report = verify_benchmark_log(
        _features(partial_posix=1), healthy, CONFIG['cleaning']
    )
    assert not passed
    assert report['checks']['module_records_complete'] == {
        'status': 'fail', 'value': 'incomplete=POSIX',
    }


# --- DATA-007: manifest lookups are exact ---------------------------------

def test_manifest_requires_exactly_one_row_per_sample():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'manifest.csv'
        cols = (
            ['benchmark', 'job_id', 'log_file', 'scenario', 'source', 'generator_label', 'note']
            + DIMENSION_NAMES + VALIDITY_COLUMNS
        )
        rows = [
            ['ior', '1', 'a.darshan', 's', 'slurm_out', 'access_granularity=1', '']
            + [1, 0, 0, 0, 0, 0, 0, 0, 0]
            + [1, 0, 0, 0, 0, 0, 0, 0, 0],
            ['ior', '2', 'b.darshan', '', 'none', '', 'no label'] + [0] * 18,
            ['custom', '3', '', 'c', 'slurm_out', 'parallelism_efficiency=1', '']
            + [0, 0, 1, 0, 0, 0, 0, 0, 0]
            + [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        m = load_manifest(path)
        assert manifest_row(m, 'ior', '1', ['/x/a.darshan'])['access_granularity'] == 1
        excluded = manifest_row(m, 'ior', '2', ['/x/b.darshan'])
        assert excluded['source'] == 'none' and excluded['note'] == 'no label'
        assert manifest_row(m, 'custom', '3', ['/x/p0.darshan', '/x/p1.darshan'])['parallelism_efficiency'] == 1
        _raises(lambda: manifest_row(m, 'ior', '9', ['/x/z.darshan']), KeyError, 'no manifest row')
        rows.append(rows[0])
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        _raises(lambda: load_manifest(path), ValueError, 'duplicate')


def _manifest_frame(rows):
    """Manifest rows as (benchmark, job_id, log_file, scenario, source, note, labels, validity)."""
    records = []
    for benchmark, job_id, log_file, scenario, source, note, labels, validity in rows:
        record = {'benchmark': benchmark, 'job_id': job_id, 'log_file': log_file,
                  'scenario': scenario, 'source': source, 'generator_label': '', 'note': note}
        record.update({d: labels.get(d, 0) for d in DIMENSION_NAMES})
        record.update({f'valid_{d}': validity.get(d, 0) for d in DIMENSION_NAMES})
        records.append(record)
    return pd.DataFrame(records)


def test_verification_report_is_an_exact_training_gate():
    from scripts.verify_all_ground_truth import report_row
    from src.data.benchmark_logs import (
        manifest_label_string, sidecar_path, write_verification_sidecar)
    healthy = {'healthy': 1}
    all_valid = {d: 1 for d in DIMENSION_NAMES}
    rows = [
        ('ior', '1', 'a.darshan', 'ior_healthy_n4', 'slurm_out', '', healthy, all_valid),
        ('ior', '2', 'b.darshan', 'ior_single_ost_n4', 'none', 'storage-layout confound', {}, {}),
        ('custom', '3', '', 'custom_balanced_n4', 'slurm_out', '',
         healthy, {'parallelism_efficiency': 1}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_path = root / 'manifest.csv'
        _manifest_frame(rows).to_csv(manifest_path, index=False)
        manifest = load_manifest(manifest_path)
        parsed = _parsed_row()
        report = [
            report_row('ior', '1', ['/x/a.darshan'], manifest_row(manifest, 'ior', '1', ['/x/a.darshan']),
                       parsed, None, CONFIG),
            report_row('ior', '2', ['/x/b.darshan'], manifest_row(manifest, 'ior', '2', ['/x/b.darshan']),
                       None, None, CONFIG),
            report_row('custom', '3', ['/x/r0.darshan', '/x/r1.darshan'],
                       manifest_row(manifest, 'custom', '3', ['/x/r0.darshan', '/x/r1.darshan']),
                       parsed, None, CONFIG),
        ]
        # the excluded sample keeps the manifest's scenario; labels follow validity
        assert report[1]['status'] == 'excluded' and report[1]['scenario'] == 'ior_single_ost_n4'
        assert report[0]['status'] == 'pass' and report[0]['labels'].endswith('healthy=1')
        assert report[2]['labels'] == 'parallelism_efficiency=0'
        assert manifest_label_string(manifest.iloc[2]) == report[2]['labels']

        report_path = root / 'verification.csv'
        pd.DataFrame(report).to_csv(report_path, index=False)
        _raises(lambda: validate_verification_report(manifest, report_path, manifest_path),
                ValueError, 'no provenance sidecar')
        write_verification_sidecar(report_path, manifest_path)
        assert validate_verification_report(manifest, report_path, manifest_path) == {
            'labeled_pass': 2, 'excluded': 1}

        # a report produced for another manifest version is refused by hash
        changed = _manifest_frame(rows)
        changed.loc[0, 'note'] = 'revised after a rerun'
        other_manifest = root / 'manifest_v2.csv'
        changed.to_csv(other_manifest, index=False)
        _raises(lambda: validate_verification_report(load_manifest(other_manifest), report_path, other_manifest),
                ValueError, 'different label_manifest')

        # a report whose label strings differ from the manifest is refused row by row
        relabeled = root / 'relabeled' / 'verification.csv'
        relabeled.parent.mkdir()
        frame = pd.DataFrame(report)
        frame.loc[0, 'labels'] = 'access_granularity=1'
        frame.to_csv(relabeled, index=False)
        write_verification_sidecar(relabeled, manifest_path)
        _raises(lambda: validate_verification_report(manifest, relabeled, manifest_path),
                ValueError, 'labels differ from the manifest')

        # a failing labeled sample blocks extraction
        failed = root / 'failed' / 'verification.csv'
        failed.parent.mkdir()
        frame = pd.DataFrame(report)
        frame.loc[0, 'status'] = 'fail'
        frame.to_csv(failed, index=False)
        write_verification_sidecar(failed, manifest_path)
        _raises(lambda: validate_verification_report(manifest, failed, manifest_path), ValueError,
                '1 labeled samples did not pass')
        assert sidecar_path(failed).name == 'verification.csv.manifest.json'


def test_chunk_merge_requires_complete_current_finite_schema():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chunks = root / 'chunks'
        lists = root / 'lists'
        chunks.mkdir()
        lists.mkdir()
        (lists / 'chunk_000.txt').write_text('/logs/a.darshan\n')
        (lists / 'chunk_001.txt').write_text('/logs/b.darshan\n')
        pd.DataFrame([_raw_row('/logs/a.darshan')]).to_parquet(
            chunks / 'chunk_000.parquet', index=False)
        pd.DataFrame([_raw_row('/logs/b.darshan')]).to_parquet(
            chunks / 'chunk_001.parquet', index=False)
        out = root / 'merged.parquet'
        result = batch_extract.merge_extraction_chunks(chunks, lists, out, 2)
        assert result == {'n_rows': 2, 'n_columns': len(_raw_row('/logs/x.darshan')),
                          'schema_version': FEATURE_SCHEMA_VERSION}
        assert sorted(pd.read_parquet(out)['_source_path']) == ['/logs/a.darshan', '/logs/b.darshan']

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chunks = root / 'chunks'
        lists = root / 'lists'
        chunks.mkdir()
        lists.mkdir()
        (lists / 'chunk_000.txt').write_text('/logs/a.darshan\n')
        bad = _raw_row('/logs/a.darshan')
        bad['_schema_version'] = 1
        bad['POSIX_READS'] = np.nan
        pd.DataFrame([bad]).to_parquet(chunks / 'chunk_000.parquet', index=False)
        _raises(lambda: batch_extract.merge_extraction_chunks(
            chunks, lists, root / 'merged.parquet', 1), batch_extract.ExtractionError,
            'schema versions')


# --- DATA-010 / DATA-011 / DATA-012: schema contracts ----------------------

def test_feature_name_api_matches_real_extraction():
    engineered = engineer_one(_parsed_row(nprocs=2, runtime=5.0), config=CONFIG)
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


def test_cleaning_drops_negative_times_of_every_layer():
    from src.data.preprocessing import CUMULATIVE_TIME_COLUMNS
    assert len(CUMULATIVE_TIME_COLUMNS) == 9
    df = raw_frame(4)
    df.loc[1, 'STDIO_F_META_TIME'] = -0.5
    df.loc[2, 'MPIIO_F_WRITE_TIME'] = -1.0
    cleaned, report = stage2_clean(df, CONFIG)
    assert report['after_timing_filter'] == 2
    assert cleaned['_jobid'].tolist() == [0, 3]


def test_config_sections_are_strict_and_the_checked_in_config_passes():
    from src.data.preprocessing import (
        CONFIG_CONTRACT, drop_excluded_features, validate_preprocessing_config)
    real = load_preprocessing_config('configs/preprocessing.yaml')
    assert set(CONFIG_CONTRACT) <= set(real)
    typo = deepcopy(real)
    typo['feature_selection']['correlation_treshold'] = typo['feature_selection'].pop('correlation_threshold')
    _raises(lambda: validate_preprocessing_config(typo), ValueError, 'correlation_threshold')
    absent = deepcopy(real)
    absent.pop('feature_exclusion')
    _raises(lambda: validate_preprocessing_config(absent), ValueError, 'feature_exclusion')
    frame = stage3_engineer(raw_frame(3), config=CONFIG)
    misspelled = {'feature_exclusion': {'drop_constant': True, 'drop_feature': []}}
    _raises(lambda: drop_excluded_features(frame, misspelled), ValueError, 'drop_features')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'preprocessing.yaml'
        path.write_text('cleaning: {min_duration_seconds: 1}\n')
        _raises(lambda: load_preprocessing_config(path), ValueError, 'cleaning keys differ')


def test_preprocessing_driver_runs_all_stages_and_samples_a_resumed_stage():
    from unittest import mock
    import scripts.run_preprocessing as driver
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / 'raw_features.parquet'
        raw_frame(30).to_parquet(raw_path, index=False)
        first = root / 'first'
        argv = ['run_preprocessing.py', '--input', str(raw_path), '--output-dir', str(first),
                '--config', 'configs/preprocessing.yaml', '--min-rows', '5']
        with mock.patch('sys.argv', argv):
            driver.main()
        manifest = json.loads((first / 'preprocessing_manifest.json').read_text())
        assert manifest['stages'] == [2, 5] and manifest['sample'] is None
        for name in ('cleaned_features.parquet', 'features.parquet', 'normalized_features.parquet',
                     'eda_stats.parquet', 'eda_correlation.parquet', 'eda_report.json',
                     'dropped_features.json', 'scalers.pkl', 'split_indices.pkl',
                     'splits/train.parquet', 'splits/val.parquet', 'splits/test.parquet'):
            assert name in manifest['artifacts'], name
        assert manifest['artifacts']['features.parquet']['rows'] == 30
        assert not list(first.glob('.*.tmp.*')) and not list((first / 'splits').glob('.*.tmp.*'))
        # a resumed stage samples the frame it loads
        second = root / 'second'
        second.mkdir()
        (second / 'features.parquet').write_bytes((first / 'features.parquet').read_bytes())
        argv = ['run_preprocessing.py', '--output-dir', str(second), '--start-stage', '5',
                '--config', 'configs/preprocessing.yaml', '--min-rows', '5', '--sample', '12']
        with mock.patch('sys.argv', argv):
            driver.main()
        resumed = json.loads((second / 'preprocessing_stage_5_5_manifest.json').read_text())
        assert resumed['sample'] == 12 and resumed['effective_min_rows'] == 5
        assert resumed['input']['path'] == str(second / 'features.parquet')
        assert resumed['artifacts']['normalized_features.parquet']['rows'] == 12
        sizes = [resumed['artifacts'][f'splits/{name}.parquet']['rows'] for name in ('train', 'val', 'test')]
        assert sum(sizes) == 12 and min(sizes) >= 1
        # a rerun refuses to replace the artifacts it wrote
        with mock.patch('sys.argv', argv):
            _raises(driver.main, FileExistsError, 'refusing to replace')


def test_missing_scaler_and_missing_config_are_errors():
    df = stage3_engineer(raw_frame(3))
    _raises(lambda: stage5_normalize(df, CONFIG, fit=False, scalers={}), ValueError, 'no fitted scaler')
    _raises(lambda: load_preprocessing_config('/nonexistent/preprocessing.yaml'), FileNotFoundError)


def test_normalization_rejects_missing_keys_and_unknown_methods():
    frame = stage3_engineer(raw_frame(3), config=CONFIG)
    missing = deepcopy(CONFIG)
    missing['normalization'].pop('rank_stat_bounded_counters')
    _raises(lambda: stage5_normalize(frame, missing), ValueError, 'missing=')
    unknown = deepcopy(CONFIG)
    unknown['normalization']['volume_counters'] = 'log1p_robst'
    _raises(lambda: stage5_normalize(frame, unknown), ValueError, 'unknown normalization')


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
    grouped = raw_frame(12)
    grouped.loc[[0, 1, 2], '_jobid'] = 10
    grouped.loc[[3, 4, 5], '_jobid'] = 20
    grouped.loc[[6, 7, 8], '_jobid'] = 30
    grouped.loc[[9, 10, 11], '_jobid'] = 40
    grouped_splits = create_splits(grouped, CONFIG)
    memberships = {}
    for name, positions in grouped_splits.items():
        for jobid in grouped['_jobid'].iloc[positions]:
            memberships.setdefault(jobid, set()).add(name)
    assert all(len(parts) == 1 for parts in memberships.values())
    unidentified = raw_frame(10).drop(columns=['_source_path'])
    unidentified['_jobid'] = 0
    _raises(lambda: create_splits(unidentified, CONFIG), ValueError,
            'cannot identify a row')


# --- DATA-016: label conversion keeps the caller's index ------------------

def test_codes_to_labels_keeps_index_and_stays_binary():
    df = stage3_engineer(raw_frame(2)).set_index(pd.Index([10, 20]))
    labels = codes_to_labels(compute_drishti_codes(df))
    assert list(labels.index) == [10, 20]
    assert labels.isin([0, 1]).all().all() and labels['healthy'].tolist() == [1, 1]


def test_shared_labels_and_drishti_baseline_remain_distinct():
    df = stage3_engineer(raw_frame(1))
    df['POSIX_WRITES'] = 2000.0
    df['POSIX_FSYNCS'] = df['POSIX_WRITES']
    shared = labels_from_features(df)
    drishti = codes_to_labels(compute_drishti_codes(df))
    assert shared['throughput_utilization'].iloc[0] == 1
    assert drishti['throughput_utilization'].iloc[0] == 0


def test_label_artifact_records_shared_rule_source():
    with tempfile.TemporaryDirectory() as tmp:
        features_path = Path(tmp) / 'features.parquet'
        labels_path = Path(tmp) / 'labels.parquet'
        features = stage3_engineer(raw_frame(2))
        features.to_parquet(features_path, index=False)
        result = generate_heuristic_labels(features_path, labels_path)
        expected = labels_from_features(features)
        pd.testing.assert_frame_equal(
            result[DIMENSION_NAMES].reset_index(drop=True),
            expected.reset_index(drop=True),
        )
        assert set(result['label_source']) == {'iosage_shared_rules'}
        manifest = json.loads((Path(str(labels_path) + '.manifest.json')).read_text())
        assert manifest['schema_version'] == 3
        assert manifest['method'] == 'iosage_shared_rules'


def test_labels_are_masked_by_target_validity_and_accepted_by_the_trainer():
    from src.models.biquality import _label_matrix, _label_validity
    rows = raw_frame(2)
    rows.loc[0, 'partial_posix'] = 1
    rows.loc[0, 'POSIX_WRITES'] = 5000.0          # random writes: access_pattern fires
    rows.loc[0, 'POSIX_SEQ_WRITES'] = 0.0
    rows.loc[1, 'POSIX_WRITES'] = 5000.0
    rows.loc[1, 'POSIX_SEQ_WRITES'] = 0.0
    engineered = stage3_engineer(rows, config=CONFIG)
    labels = labels_from_features(engineered)
    validity = validity_from_features(engineered)
    assert labels.loc[1, 'access_pattern'] == 1 and labels.loc[1, 'healthy'] == 0
    assert validity.loc[0, 'access_pattern'] == 0 and labels.loc[0, 'access_pattern'] == 0
    assert labels.loc[0, 'healthy'] == 1 and validity.loc[0, 'healthy'] == 0
    with tempfile.TemporaryDirectory() as tmp:
        features_path = Path(tmp) / 'features.parquet'
        labels_path = Path(tmp) / 'labels.parquet'
        engineered.to_parquet(features_path, index=False)
        written = generate_heuristic_labels(features_path, labels_path)
    _label_matrix(written, 'production labels')
    valid = _label_validity(written, 'production labels', required=True)
    assert valid.tolist() == [[False, False, False, False, False, True, False, False],
                              [True] * 8]


def test_label_artifact_refuses_empty_features():
    with tempfile.TemporaryDirectory() as tmp:
        features_path = Path(tmp) / 'features.parquet'
        labels_path = Path(tmp) / 'labels.parquet'
        stage3_engineer(raw_frame(1)).iloc[:0].to_parquet(features_path, index=False)
        _raises(
            lambda: generate_heuristic_labels(features_path, labels_path),
            ValueError,
            'no rows',
        )
        assert not labels_path.exists()
