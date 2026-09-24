"""Unit tests for the per-rank statistics and the rank-imbalance rule (D8)."""
import pandas as pd

from src.data.benchmark_verify import rank_imbalance_present
from src.data.feature_extraction import compute_layer_and_rank_features
from src.data.parse_darshan import (
    RANK_STAT_KEYS, _extract_pydarshan_module, _rank_statistics, parse_benchmark_job)
from src.data.preprocessing import stage3_engineer
from tests.pipeline_fixtures import raw_frame

_NAMES = {1: '/scratch/a.dat', 2: '/scratch/b.dat', 3: '<STDERR>'}


def _posix(rows):
    """rows: (id, rank, bytes_read, bytes_written, read_time, write_time, meta_time)."""
    ints = pd.DataFrame([{'id': r[0], 'rank': r[1], 'POSIX_BYTES_READ': r[2],
                          'POSIX_BYTES_WRITTEN': r[3]} for r in rows])
    floats = pd.DataFrame([{'id': r[0], 'rank': r[1], 'POSIX_F_READ_TIME': r[4],
                            'POSIX_F_WRITE_TIME': r[5], 'POSIX_F_META_TIME': r[6]}
                           for r in rows])
    return {'POSIX': {'counters': ints, 'fcounters': floats}}


def _features(stats, nprocs, bytes_all, time_all=1.0):
    """Derived features from the statistics, the way stage 3 sees them."""
    raw = dict(stats)
    raw.update({'POSIX_BYTES_READ': 0.0, 'POSIX_BYTES_WRITTEN': bytes_all,
                'STDIO_BYTES_READ': 0.0, 'STDIO_BYTES_WRITTEN': 0.0,
                'POSIX_READS': 0.0, 'POSIX_WRITES': 1.0, 'STDIO_READS': 0.0, 'STDIO_WRITES': 0.0,
                'POSIX_F_READ_TIME': 0.0, 'POSIX_F_WRITE_TIME': time_all, 'POSIX_F_META_TIME': 0.0,
                'STDIO_F_READ_TIME': 0.0, 'STDIO_F_WRITE_TIME': 0.0, 'STDIO_F_META_TIME': 0.0})
    out = compute_layer_and_rank_features(lambda k: float(raw.get(k, 0.0)), nprocs)
    out['nprocs'] = nprocs
    out['SHARED_BYTE_IMBALANCE'] = stats['SHARED_BYTE_IMBALANCE']
    return out


def test_file_per_process_imbalance_is_visible():
    # rank 0 writes 10, ranks 1..3 write 1 each: file per process, no shared record
    dfs = _posix([(1, 0, 0, 10, 0, 1.0, 0), (2, 1, 0, 1, 0, 0.1, 0),
                  (2, 2, 0, 1, 0, 0.1, 0), (2, 3, 0, 1, 0, 0.1, 0)])
    s = _rank_statistics(dfs, 4, _NAMES)
    assert set(s) == set(RANK_STAT_KEYS)
    assert s['RANK_IO_COUNT'] == 4 and s['RANK_BYTES_MAX'] == 10 and s['RANK_BYTES_MIN'] == 1
    assert s['RANK_SHARED_BYTES'] == 0
    f = _features(s, 4, 13.0, 1.3)
    assert abs(f['top_rank_byte_share'] - 10 / 13) < 1e-9
    assert abs(f['top_rank_time_share'] - 1.0 / 1.3) < 1e-9
    assert abs(f['rank_byte_range_ratio'] - 0.9) < 1e-9
    assert abs(f['rank_time_range_ratio'] - 0.9) < 1e-9
    assert f['rank_bytes_cv_all'] > 1.0 and s['RANK_BYTES_GINI'] > 0.4
    assert rank_imbalance_present(f)


def test_balanced_ranks_have_zero_spread():
    dfs = _posix([(1, r, 0, 5, 0, 0.5, 0) for r in range(4)])
    f = _features(_rank_statistics(dfs, 4, _NAMES), 4, 20.0, 2.0)
    assert f['rank_bytes_cv_all'] == 0 and f['top_rank_byte_share'] == 0.25
    assert f['rank_byte_range_ratio'] == 0
    assert not rank_imbalance_present(f)


def test_shared_record_is_spread_evenly_and_counted():
    # one shared record (rank -1) of 8 bytes plus rank 0 writing 4 of its own
    dfs = _posix([(1, -1, 0, 8, 0, 0.8, 0), (2, 0, 0, 4, 0, 0.4, 0)])
    s = _rank_statistics(dfs, 4, _NAMES)
    assert s['RANK_SHARED_BYTES'] == 8 and s['RANK_BYTES_MAX'] == 6 and s['RANK_IO_COUNT'] == 4
    f = _features(s, 4, 12.0)
    assert abs(f['shared_record_byte_share'] - 8 / 12) < 1e-9


def test_single_writer_in_a_large_job():
    dfs = _posix([(1, 0, 0, 100, 0, 1.0, 0)])
    s = _rank_statistics(dfs, 32, _NAMES)
    assert s['RANK_IO_COUNT'] == 1 and s['RANK_BYTES_MIN'] == 0
    assert abs(s['RANK_BYTES_GINI'] - (1 - 1 / 32)) < 1e-9
    f = _features(s, 32, 100.0)
    assert f['top_rank_byte_share'] == 1.0 and abs(f['io_rank_fraction'] - 1 / 32) < 1e-9
    assert f['rank_byte_range_ratio'] == 1.0 and rank_imbalance_present(f)


def test_per_file_imbalance_follows_drishti_and_skips_standard_streams():
    # file 1 written by ranks 0 and 1 (100 vs 40): (100 - 40) / 100 = 0.6
    # <STDERR> written unevenly by all ranks must not count
    dfs = _posix([(1, 0, 0, 100, 0, 0, 0), (1, 1, 0, 40, 0, 0, 0),
                  (3, 0, 0, 300, 0, 0, 0), (3, 1, 0, 10, 0, 0, 0)])
    s = _rank_statistics(dfs, 2, _NAMES)
    assert abs(s['FILE_WRITE_IMBALANCE'] - 0.6) < 1e-9 and s['FILE_READ_IMBALANCE'] == 0
    # a job whose only writes are log messages has no I/O ranks at all
    only_streams = _posix([(3, 0, 0, 300, 0, 0, 0), (3, 1, 0, 10, 0, 0, 0)])
    s = _rank_statistics(only_streams, 2, _NAMES)
    assert s['FILE_WRITE_IMBALANCE'] == 0 and s['RANK_IO_COUNT'] == 0 and s['RANK_BYTES_MAX'] == 0


def test_shared_record_straggler_evidence_survives():
    # one file opened by all four ranks: Darshan keeps fastest/slowest bytes
    ints = pd.DataFrame([{'id': 1, 'rank': -1, 'POSIX_BYTES_READ': 0, 'POSIX_BYTES_WRITTEN': 1000,
                          'POSIX_FASTEST_RANK_BYTES': 50, 'POSIX_SLOWEST_RANK_BYTES': 700}])
    floats = pd.DataFrame([{'id': 1, 'rank': -1, 'POSIX_F_READ_TIME': 0.0, 'POSIX_F_WRITE_TIME': 10.0,
                            'POSIX_F_META_TIME': 0.0, 'POSIX_F_FASTEST_RANK_TIME': 1.0,
                            'POSIX_F_SLOWEST_RANK_TIME': 6.0}])
    s = _rank_statistics({'POSIX': {'counters': ints, 'fcounters': floats}}, 4, _NAMES)
    assert abs(s['SHARED_BYTE_IMBALANCE'] - 0.65) < 1e-9
    assert abs(s['SHARED_TIME_IMBALANCE'] - 0.5) < 1e-9
    f = _features(s, 4, 1000.0)
    assert f['rank_byte_range_ratio'] == 0 and rank_imbalance_present(f)


def test_record_beyond_nprocs_is_an_error():
    dfs = _posix([(1, 5, 0, 1, 0, 0, 0)])
    try:
        _rank_statistics(dfs, 4, _NAMES)
    except ValueError as exc:
        assert 'rank 5' in str(exc)
    else:
        raise AssertionError('expected ValueError')


_RANK_INT = {'POSIX_FASTEST_RANK': 1, 'POSIX_FASTEST_RANK_BYTES': 2,
             'POSIX_SLOWEST_RANK': 3, 'POSIX_SLOWEST_RANK_BYTES': 6}
_RANK_FLOAT = {'POSIX_F_FASTEST_RANK_TIME': 0.1, 'POSIX_F_SLOWEST_RANK_TIME': 0.9,
               'POSIX_F_VARIANCE_RANK_BYTES': 2.5, 'POSIX_F_VARIANCE_RANK_TIME': 0.5}


def _posix_with_variance(ids, rank=-1):
    ints = pd.DataFrame([dict({'id': i, 'rank': rank, 'POSIX_BYTES_READ': 0, 'POSIX_BYTES_WRITTEN': 8},
                              **_RANK_INT) for i in ids])
    floats = pd.DataFrame([dict({'id': i, 'rank': rank, 'POSIX_F_READ_TIME': 0.0,
                                 'POSIX_F_WRITE_TIME': 1.0, 'POSIX_F_META_TIME': 0.0}, **_RANK_FLOAT)
                           for i in ids])
    return {'counters': ints, 'fcounters': floats}


def test_shared_indicator_covers_any_shared_file_but_variance_needs_one_file():
    counters = {}
    assert _extract_pydarshan_module(_posix_with_variance([1]), 'POSIX', counters, 4)
    assert counters['POSIX_F_VARIANCE_RANK_BYTES'] == 2.5
    counters = {}
    assert _extract_pydarshan_module(_posix_with_variance([1, 2]), 'POSIX', counters, 4)
    assert counters['POSIX_F_VARIANCE_RANK_BYTES'] == 0.0
    # one private record from rank 3 is not a shared file
    counters = {}
    assert not _extract_pydarshan_module(_posix_with_variance([99], rank=3), 'POSIX', counters, 4)
    assert counters['POSIX_F_VARIANCE_RANK_BYTES'] == 0.0 and counters['POSIX_FASTEST_RANK'] == -1.0


def test_merged_per_rank_records_of_one_file_get_darshan_reduction():
    # four processes, one file, per-rank records: rank 2 slowest, rank 0 fastest
    rows = [(0, 0.5, 100), (1, 1.0, 100), (2, 4.0, 400), (3, 2.0, 100)]
    ints = pd.DataFrame([dict({'id': 7, 'rank': r, 'POSIX_BYTES_READ': 0, 'POSIX_BYTES_WRITTEN': b},
                              **{k: 0 for k in _RANK_INT}) for r, _, b in rows])
    floats = pd.DataFrame([dict({'id': 7, 'rank': r, 'POSIX_F_READ_TIME': 0.0, 'POSIX_F_WRITE_TIME': t,
                                 'POSIX_F_META_TIME': 0.0}, **{k: 0.0 for k in _RANK_FLOAT}) for r, t, _ in rows])
    counters = {}
    assert _extract_pydarshan_module({'counters': ints, 'fcounters': floats}, 'POSIX', counters, 4)
    assert counters['POSIX_SLOWEST_RANK'] == 2 and counters['POSIX_SLOWEST_RANK_BYTES'] == 400
    assert counters['POSIX_FASTEST_RANK'] == 0 and counters['POSIX_F_FASTEST_RANK_TIME'] == 0.5
    assert abs(counters['POSIX_F_VARIANCE_RANK_BYTES'] - 16875.0) < 1e-6


def test_max_time_size_uses_row_position_not_index_label():
    ints = pd.DataFrame([{'id': 1, 'rank': 0, 'POSIX_MAX_WRITE_TIME_SIZE': 10},
                         {'id': 2, 'rank': 1, 'POSIX_MAX_WRITE_TIME_SIZE': 20}], index=[5, 7])
    floats = pd.DataFrame([{'id': 1, 'rank': 0, 'POSIX_F_MAX_WRITE_TIME': 1.0},
                           {'id': 2, 'rank': 1, 'POSIX_F_MAX_WRITE_TIME': 3.0}], index=[5, 7])
    counters = {}
    _extract_pydarshan_module({'counters': ints, 'fcounters': floats}, 'POSIX', counters, 2)
    assert counters['POSIX_MAX_WRITE_TIME_SIZE'] == 20


def test_merge_refuses_unreadable_logs():
    for bad in ([], ['/nonexistent/rank0.darshan']):
        try:
            parse_benchmark_job(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError for %r' % (bad,))


def test_dict_and_vectorized_paths_agree():
    raw = dict(raw_frame(1).iloc[0])
    raw.update({'nprocs': 4, 'runtime_seconds': 10.0, 'POSIX_BYTES_READ': 0.0,
           'POSIX_BYTES_WRITTEN': 13.0, 'STDIO_BYTES_READ': 0.0, 'STDIO_BYTES_WRITTEN': 3.0,
           'POSIX_READS': 0.0, 'POSIX_WRITES': 4.0, 'STDIO_READS': 0.0, 'STDIO_WRITES': 1.0,
           'POSIX_F_READ_TIME': 0.0, 'POSIX_F_WRITE_TIME': 1.0, 'POSIX_F_META_TIME': 0.2,
           'STDIO_F_READ_TIME': 0.0, 'STDIO_F_WRITE_TIME': 0.1, 'STDIO_F_META_TIME': 0.0,
           'RANK_IO_COUNT': 4.0, 'RANK_BYTES_MAX': 10.0, 'RANK_BYTES_MIN': 1.0,
           'RANK_BYTES_VAR': 15.1875, 'RANK_BYTES_GINI': 0.5, 'RANK_TIME_MAX': 1.0,
           'RANK_TIME_MIN': 0.1, 'RANK_TIME_VAR': 0.15,
           'RANK_BYTES_TOTAL': 16.0, 'RANK_TIME_TOTAL': 1.3,
           'RANK_SHARED_BYTES': 0.0, 'FILE_WRITE_IMBALANCE': 0.0, 'FILE_READ_IMBALANCE': 0.0})
    scalar = compute_layer_and_rank_features(lambda k: float(raw[k]), 4)
    df = pd.DataFrame([raw])
    vector = compute_layer_and_rank_features(lambda k: df[k], df['nprocs'])
    for name, value in scalar.items():
        assert abs(float(vector[name].iloc[0]) - value) < 1e-9, name
    assert abs(scalar['io_bytes_all'] - 16.0) < 1e-9
    assert abs(scalar['stdio_byte_share'] - 3 / 16) < 1e-9
    assert abs(scalar['avg_write_size_all'] - 16 / 5) < 1e-9
    engineered = stage3_engineer(df)
    assert abs(engineered['top_rank_byte_share'].iloc[0] - 10 / 16) < 1e-9
    assert abs(engineered['rank_byte_range_ratio'].iloc[0] - 0.9) < 1e-9


def test_rank_imbalance_rule_is_drishti_size_threshold():
    # one process is never imbalanced; otherwise (max - min) / max > 0.3,
    # so a rank doing twice the others (0.5) is flagged at any rank count
    def f(nprocs, ratio, shared=0.0):
        return {'nprocs': nprocs, 'rank_byte_range_ratio': ratio, 'SHARED_BYTE_IMBALANCE': shared}
    assert not rank_imbalance_present(f(1, 1.0))
    assert rank_imbalance_present(f(64, 0.5))
    assert not rank_imbalance_present(f(4, 0.3))
    assert not rank_imbalance_present(f(4, 0.0))
    # a shared file whose slowest rank moved 20 percent more than the fastest
    assert rank_imbalance_present(f(4, 0.0, shared=0.2)) and not rank_imbalance_present(f(4, 0.0, shared=0.15))
