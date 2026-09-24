"""submit_and_wait passes sbatch options before the script and positional arguments after it.

The application run scripts use --export=NONE, so their case identity travels in an
--export option; their knobs are positional arguments. Both must reach sbatch in order,
and the default call (script only) must be unchanged.
"""
import importlib
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class _Result:
    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _run_with_fake_sbatch(monkeypatch_calls, **kwargs):
    mod = importlib.import_module("src.llm.iterative_executor")
    ex = mod.IterativeExecutor.__new__(mod.IterativeExecutor)
    calls = []

    def fake_run(cmd, **_):
        calls.append(cmd)
        if cmd[0] == "sbatch":
            return _Result("Submitted batch job 4242\n")
        return _Result("4242|COMPLETED\n4242.batch|COMPLETED\n")

    original = mod.subprocess.run
    mod.subprocess.run = fake_run
    try:
        job = ex.submit_and_wait("/x/run.slurm", poll_interval=0, **kwargs)
    finally:
        mod.subprocess.run = original
    monkeypatch_calls.extend(calls)
    return job


def test_default_call_is_script_only():
    calls = []
    assert _run_with_fake_sbatch(calls) == "4242"
    assert calls[0] == ["sbatch", "/x/run.slurm"], calls[0]


def test_options_before_and_arguments_after_the_script():
    calls = []
    job = _run_with_fake_sbatch(calls, sbatch_args=["--export=CASE_ID=N1,REPEAT=3", "--time=00:10:00"],
                                script_args=["50", "25"])
    assert job == "4242"
    assert calls[0] == ["sbatch", "--export=CASE_ID=N1,REPEAT=3", "--time=00:10:00",
                        "/x/run.slurm", "50", "25"], calls[0]


if __name__ == "__main__":
    test_default_call_is_script_only()
    test_options_before_and_arguments_after_the_script()
    print("2/2 submit-argument tests pass")
