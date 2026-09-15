"""Unit tests for the closed-loop objective (A1): wall time + work guard + repeats."""
from src.llm.closed_loop_metrics import (
    phase_walltime, aggregate_repeats, evaluate_candidate, work_params_changed,
    configured_work, select_primary_log)


def _m(wall, bw=10.0, bytes_total=1e9, cfg=None):
    return {"walltime_s": wall, "write_bw_mb_s": bw, "bytes_total": bytes_total,
            "configured_bytes": cfg, "spread_rel": 0.0, "n_repeats": 1, "n_phases": 1}


def test_phase_walltime_concurrent_and_sequential():
    # two concurrent per-rank logs (one phase) then a later phase
    s = [{"start": 0, "runtime": 10.0}, {"start": 1, "runtime": 12.0}, {"start": 30, "runtime": 5.0}]
    wall, n = phase_walltime(s)
    # phase 1: logs (0,10) and (1,12) overlap -> longest = 12; phase 2: 5 -> 17 total
    assert n == 2 and abs(wall - 17.0) < 1e-9


def test_aggregate_repeats_median_and_spread():
    agg = aggregate_repeats([_m(10.0), _m(12.0), _m(11.0)])
    assert agg["walltime_s"] == 11.0 and agg["n_repeats"] == 3
    assert abs(agg["spread_rel"] - 2.0 / 11.0) < 1e-9


def test_walltime_speedup_accepts_fewer_bytes():
    # E2E NOFILL: fewer bytes moved, same configured work -> accepted on wall time
    base = aggregate_repeats([_m(19.31, bw=32.7, bytes_total=1.57e9, cfg=100)])
    new = aggregate_repeats([_m(2.98, bw=11.3, bytes_total=0.67e9, cfg=100)])
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert d["accepted"] and not d["regression"] and abs(d["speedup"] - 6.48) < 0.01
    assert d["bw_speedup"] < 1.0 and d["bytes_ratio_flag"] and not d["rejected_work_changed"]


def test_configured_work_change_is_rejected():
    base = aggregate_repeats([_m(100.0, cfg=1000)])
    new = aggregate_repeats([_m(1.0, cfg=10)])          # 100x "speedup" by doing 1% of the work
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert d["rejected_work_changed"] and d["regression"] and not d["accepted"]


def test_noise_margin_blocks_small_gains():
    base = aggregate_repeats([_m(10.0), _m(11.0), _m(12.0)])   # spread 18%
    new = aggregate_repeats([_m(10.2), _m(10.4), _m(10.3)])    # 6.8% better than the median
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert not d["accepted"] and not d["regression"]


def test_regression_rollback():
    d = evaluate_candidate(aggregate_repeats([_m(10.0)]), aggregate_repeats([_m(12.0)]), 1.0)
    assert d["regression"] and not d["accepted"]


def test_work_params_changed_ior_sizes_and_mdtest():
    assert work_params_changed("ior", {"block_size": "1m", "segments": 4, "transfer_size": "64"},
                               {"block_size": "1024k", "segments": 4, "transfer_size": "1m"}) == []
    assert work_params_changed("mdtest", {"items_per_rank": 10000, "write_bytes": 4096},
                               {"items_per_rank": 100, "write_bytes": 4096}) == ["items_per_rank"]
    assert work_params_changed("dlio", {"computation_time": 0.1}, {"computation_time": 0.0}) == ["computation_time"]


def test_configured_work_ratio_is_unit_free():
    b, _ = configured_work("ior_small_posix", {"block_size": "1m", "segments": 2}, 64, "ior")
    n, _ = configured_work("ior_small_posix", {"block_size": "1024k", "segments": 2}, 64, "ior")
    assert b == n and b > 0
    m, files = configured_work("mdtest_x", {"items_per_rank": 10, "write_bytes": 0}, 4, "mdtest")
    assert files == 40 and m == 40


def test_primary_log_selection():
    s = [{"name": "h5bench_read_x", "start": 20, "bytes_written": 0, "bytes_read": 5},
         {"name": "h5bench_write_x", "start": 0, "bytes_written": 5, "bytes_read": 0}]
    assert select_primary_log(s, "h5bench")["name"] == "h5bench_write_x"
    assert select_primary_log(s, "dlio")["name"] in ("h5bench_read_x", "h5bench_write_x")
