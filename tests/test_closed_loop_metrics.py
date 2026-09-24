"""Unit tests for the closed-loop objective (A1): wall time + work guard + repeats."""
from src.llm.closed_loop_metrics import (
    phase_walltime, aggregate_repeats, evaluate_candidate, work_params_changed,
    configured_work, select_primary_log, median_ci, _rel_mad)


def _m(wall, bw=10.0, bytes_total=1e9, cfg=None):
    return {"walltime_s": wall, "write_bw_mb_s": bw, "bytes_total": bytes_total,
            "configured_bytes": cfg, "spread_rel": 0.0, "n_repeats": 1, "n_phases": 1}


def test_phase_walltime_concurrent_and_sequential():
    # two concurrent per-rank logs (one phase) then a later phase
    s = [{"start": 0, "runtime": 10.0}, {"start": 1, "runtime": 12.0}, {"start": 30, "runtime": 5.0}]
    wall, n = phase_walltime(s)
    # phase 1 is the union [0, 13]; phase 2 is [30, 35].
    assert n == 2 and abs(wall - 18.0) < 1e-9
    staggered = [{"start": 0, "runtime": 10.0}, {"start": 9, "runtime": 10.0}]
    assert phase_walltime(staggered) == (19.0, 1)
    chained = [{"start": 0, "runtime": 3.0}, {"start": 3, "runtime": 4.0},
               {"start": 6, "runtime": 3.0}]
    assert phase_walltime(chained) == (9.0, 1)
    nested = [{"start": 0, "runtime": 10.0}, {"start": 2, "runtime": 1.0}]
    assert phase_walltime(nested) == (10.0, 1)


def test_aggregate_repeats_median_and_spread():
    agg = aggregate_repeats([_m(10.0), _m(12.0), _m(11.0)])
    assert agg["walltime_s"] == 11.0 and agg["n_repeats"] == 3
    assert abs(agg["spread_rel"] - 2.0 / 11.0) < 1e-9


def _runs(values, **kw):
    return aggregate_repeats([_m(v, **kw) for v in values])


# eight runs with realistic jitter around a centre
_J = [0.97, 1.03, 0.99, 1.01, 0.95, 1.05, 1.00, 1.02]


def test_median_ci_order_statistics():
    ci = median_ci(list(range(1, 9)), confidence=0.90)       # n=8 -> 2nd and 7th, coverage 92.97%
    assert (ci["lower"], ci["upper"], ci["rank"]) == (2, 7, 2) and ci["valid"]
    assert abs(ci["coverage"] - 0.9297) < 1e-3
    assert not median_ci([1, 2, 3], confidence=0.90)["valid"]  # 3 runs reach only 75%
    ci6 = median_ci([1, 2, 3, 4, 5, 6], confidence=0.95)      # n=6 -> min/max, coverage 96.9%
    assert ci6["valid"] and ci6["rank"] == 1


def test_walltime_speedup_accepts_fewer_bytes():
    # E2E NOFILL: fewer bytes moved, same configured work -> accepted on wall time
    base = _runs([19.31 * j for j in _J], bw=32.7, bytes_total=1.57e9, cfg=100)
    new = _runs([2.98 * j for j in _J], bw=11.3, bytes_total=0.67e9, cfg=100)
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert d["accepted"] and d["verdict"] == "faster" and not d["regression"]
    assert abs(d["speedup"] - 6.48) < 0.05
    assert d["bw_speedup"] < 1.0 and d["bytes_ratio_flag"] and not d["rejected_work_changed"]
    assert d["speedup_ci"][0] > 5.0 and d["speedup_ci"][1] < 8.0


def test_configured_work_change_is_rejected():
    base = _runs([100.0 * j for j in _J], cfg=1000)
    new = _runs([1.0 * j for j in _J], cfg=10)          # 100x "speedup" by doing 1% of the work
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert d["rejected_work_changed"] and d["regression"] and not d["accepted"]
    assert d["verdict"] == "rejected_work_changed"


def test_overlapping_intervals_mean_no_significant_change():
    base = _runs([10.0 * j for j in _J])
    new = _runs([9.7 * j for j in _J])                   # 3% better, inside the noise
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert not d["accepted"] and not d["regression"] and d["verdict"] == "no_significant_change"


def test_null_change_is_never_a_win():
    # A/A comparison: the same configuration measured twice must not be accepted.
    a = _runs([17.0, 18.3, 15.3, 18.2, 16.2, 20.6, 13.8, 17.5])
    b = _runs([16.8, 17.9, 15.9, 18.0, 16.0, 19.9, 14.2, 17.1])
    d = evaluate_candidate(a, b, best_speedup=1.0)
    assert not d["accepted"] and d["verdict"] == "no_significant_change"


def test_real_gain_survives_a_burst_outlier():
    # The measured fsync case: a true ~2x gain with one 5x burst in the baseline. The old
    # range-based margin (363%) rejected it; the median's interval ignores the burst.
    base = _runs([507.6, 84.1, 116.6, 94.5, 102.0, 98.0, 110.0, 91.0])
    new = _runs([57.7, 35.4, 35.0, 47.7, 52.0, 44.0, 49.0, 41.0])
    d = evaluate_candidate(base, new, best_speedup=1.0)
    assert d["accepted"] and d["verdict"] == "faster", d
    assert base["spread_rel"] > 3.0 and base["rel_mad"] < 0.2


def test_too_few_runs_decide_nothing():
    d = evaluate_candidate(_runs([10.0, 11.0, 12.0]), _runs([5.0, 5.1, 5.2]), 1.0)
    assert not d["accepted"] and d["verdict"] == "insufficient_runs"
    assert d["decision_basis"] == "insufficient_runs" and d["speedup_ci"] is None


def test_regression_needs_separated_intervals():
    d = evaluate_candidate(_runs([10.0 * j for j in _J]), _runs([14.0 * j for j in _J]), 1.0)
    assert d["regression"] and d["verdict"] == "slower" and not d["accepted"]


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


def test_rel_mad_ignores_a_single_outlier():
    runs = [20.0, 21.0, 19.0, 61.0]     # the 61 s run is the shared-filesystem outlier
    assert _rel_mad(runs) < 0.2, _rel_mad(runs)
    assert (max(runs) - min(runs)) / 20.5 > 2.0   # the range-based estimate explodes
