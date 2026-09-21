"""Regression tests for repeated execution of one configuration.

The executor derives the per-job scratch directory from the job name, and the benchmark
command already carries that path, so every repeat must reuse the same job name.
"""
import importlib
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load_optimizer():
    # Import as part of the package: the module uses relative imports.
    return importlib.import_module("src.llm.iterative_optimizer")


class _FakeExecutor:
    """Records the job names used and mimics the per-job scratch directory rule."""

    def __init__(self, scratch_dir="/scratch"):
        self.scratch_dir = scratch_dir
        self.calls = []

    def execute_benchmark(self, cmd, **kwargs):
        self.calls.append(kwargs["job_name"])
        job_scratch = f"{self.scratch_dir}/{kwargs['job_name']}"
        # The benchmark writes where cmd says; the job only creates its own scratch dir.
        if job_scratch not in cmd:
            return {"success": False, "job_id": None, "error": "ENOENT"}
        return {"success": True, "job_id": f"job{len(self.calls)}", "features": {},
                "metrics": {"total_bw_mb_s": 100.0}, "darshan_paths": [],
                "measurement": {"walltime_s": 10.0 + len(self.calls), "nprocs": 16,
                                "write_bw_mb_s": 100.0, "bytes_total": 1, "n_phases": 1,
                                "configured_bytes": 1}}


def test_repeats_reuse_the_job_name_and_scratch_directory():
    mod = _load_optimizer()
    opt = mod.IterativeOptimizer.__new__(mod.IterativeOptimizer)
    opt.executor = _FakeExecutor()
    opt.iter_config = {"iteration": {"confidence": 0.90}}
    cmd = "ior -o /scratch/wl_baseline/ior_test_file"
    first, agg, runs = opt._execute_repeated(cmd, {"job_name": "wl_baseline"}, 3)
    assert opt.executor.calls == ["wl_baseline"] * 3, opt.executor.calls
    assert len(runs) == 3, runs
    assert first is not None and agg["n_repeats"] == 3
    assert agg["walltime_s"] == 12.0, agg["walltime_s"]


def test_failed_repeats_are_skipped_not_counted():
    mod = _load_optimizer()
    opt = mod.IterativeOptimizer.__new__(mod.IterativeOptimizer)
    opt.executor = _FakeExecutor()
    opt.iter_config = {"iteration": {"confidence": 0.90}}
    first, agg, runs = opt._execute_repeated("ior -o /elsewhere/ior_test_file",
                                             {"job_name": "wl_baseline"}, 3)
    assert first is None and runs == [] and not agg


def test_interleaved_runs_alternate_control_and_candidate():
    mod = _load_optimizer()
    opt = mod.IterativeOptimizer.__new__(mod.IterativeOptimizer)
    opt.executor = _FakeExecutor()
    opt.iter_config = {"iteration": {"confidence": 0.90}}
    first, cand, ctrl = opt._execute_interleaved(
        "ior -o /scratch/wl_baseline/f", {"job_name": "wl_baseline"},
        "ior -o /scratch/wl_i0/f", {"job_name": "wl_i0"}, 8)
    assert opt.executor.calls == ["wl_baseline", "wl_i0"] * 8, opt.executor.calls
    assert cand["n_repeats"] == 8 and ctrl["n_repeats"] == 8
    assert cand["ci_valid"] and ctrl["ci_valid"] and first is not None


if __name__ == "__main__":
    test_repeats_reuse_the_job_name_and_scratch_directory()
    test_failed_repeats_are_skipped_not_counted()
    test_interleaved_runs_alternate_control_and_candidate()
    print("3/3 repeated-execution tests pass")
