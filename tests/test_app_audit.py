"""Audit of a synthetic application series against the Nek5000 preregistration."""
import importlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

audit = importlib.import_module("scripts.apps.audit_app_runs")
CASES = PROJECT_DIR / "configs" / "app_cases.yaml"
WORK = {"steps": 200, "checkpoint_interval": 50, "expected_checkpoints": 5, "precision": "double", "ranks": 32,
        "nodes": 1, "mesh_sha256": "m", "input_sha256": "i"}
IO = {"N1": {"files_per_checkpoint": 1, "write_module": "MPI-IO", "shared_file": True},
      "N2": {"files_per_checkpoint": 32, "write_module": "STDIO", "shared_file": False}}


COMPONENTS = ["X0", "X1", "X2", "U0", "U1", "U2", "P0", "X_minmax0", "U_minmax0", "P_minmax0"]


def _result(final_time=5.05374):
    checkpoints = {}
    for step in (0, 50, 100, 150, 200):
        checkpoints[f"step_{step}"] = {"istep": step, "nelog": 1536, "varcode": "XUP", "nxyz": 512, "wdsize": 8,
                                       "variables": {c: {"sum": 1000.0 + step, "min": -1.0, "max": 1.0, "count": 786432}
                                                     for c in COMPONENTS}}
    return {"final_step": 200, "run_successful": True, "volflow_step": 200, "final_time": final_time,
            "final_dt": 0.0249373, "final_courant": 2.962, "volflow": [5.0537, 0.0037109, 0.00037575, 6.2828, 6.2832],
            "checkpoints": checkpoints}


def _row(case, repeat, wall, control=False, start=0):
    role = "problem" if case == "N2" else "fix"
    return {"app": "nek5000", "attempt": "t", "case": case, "role": role, "kind": "constructed", "repeat": repeat,
            "jobid": f"{case}{repeat}{'c' if control else ''}", "nodes": 1, "ntasks": 32, "nodelist": "cn1",
            "start_iso": f"2026-09-24T00:{start:02d}:00-05:00", "wall_s": wall, "rc": 0, "nodarshan": control,
            "app_metric": {"name": "total_elapsed_s", "value": wall - 1}, "knobs": {"writeNFiles": 32 if case == "N2" else 1},
            "work": dict(WORK), "correctness": {"check": "nek5000_log_and_checkpoint_fields", "pass": True, "result": _result()},
            "io_validation": {"check": "control_run_without_darshan" if control else "per_file_darshan_bytes_and_organization",
                              "pass": True, "expected": dict(IO[case]), "observed": {"files_per_checkpoint": [IO[case]["files_per_checkpoint"]] * 5},
                              "modules": [] if control else ["POSIX", "STDIO"]},
            "out_files": 5 * IO[case]["files_per_checkpoint"], "out_bytes": 220662440, "scratch_root": "/s",
            "darshan_logs": [] if control else [f"/d/{case}{repeat}.darshan"], "binary_sha256": "b",
            "script": "/s.slurm", "script_sha256": "s", "iosage_commit": "c"}


def _series(repeats=11, n2_wall=40.0, n1_wall=33.0):
    rows = []
    minute = 0
    for k in range(repeats):
        for case, wall in (("N2", n2_wall), ("N1", n1_wall)):
            rows.append(_row(case, k, wall + 0.1 * k, start=minute)); minute += 1
    for case, wall in (("N2", n2_wall), ("N1", n1_wall)):
        rows.append(_row(case, 0, wall, control=True, start=minute)); minute += 1
    return rows


def _run(rows, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp, "manifest.jsonl")
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = Path(tmp, "audit.json")
        argv = ["audit", "--manifest", str(manifest), "--cases", str(CASES), "--app", "nek5000", "--output", str(out), *extra]
        original_argv, original_summary = sys.argv, audit.darshan_summary
        sys.argv = argv
        audit.darshan_summary = lambda path: {"written": 0, "modules": ["POSIX", "STDIO"], "partial": [], "nprocs": 32}
        try:
            audit.main()
        except SystemExit as exc:
            code = exc.code
        finally:
            sys.argv, audit.darshan_summary = original_argv, original_summary
        return code, json.loads(out.read_text())


def test_valid_series_passes_and_reports_the_verdict_as_produced():
    code, report = _run(_series())
    assert code == 0, report["violations"]
    assert report["repeats_required"] == 11 and report["verdicts"]["N2:N1"]["verdict"] == "faster"
    assert report["classifier_support"] == {
        "N1": {"supported": False, "expected_labels": [], "allowed_extra_labels": []},
        "N2": {"supported": False, "expected_labels": [], "allowed_extra_labels": []},
    }
    code, report = _run(_series(n2_wall=33.0, n1_wall=33.0))
    assert code == 0 and report["verdicts"]["N2:N1"]["verdict"] == "no_significant_change", report["violations"]
    code, report = _run(_series(n2_wall=33.0, n1_wall=33.0), extra=["--pair", "N2:N1:faster"])
    assert code == 1 and any("required faster" in v for v in report["violations"])


def test_protocol_and_evidence_violations_are_reported():
    rows = _series(); rows.pop(3)
    code, report = _run(rows)
    assert code == 1 and any("exactly 11 required" in v for v in report["violations"])
    rows = _series(); rows[2]["work"]["steps"] = 100
    code, report = _run(rows)
    assert any("work.steps" in v for v in report["violations"]) and any("differ across" in v for v in report["violations"])
    rows = _series(); rows[5]["correctness"]["result"]["final_time"] = 5.06
    code, report = _run(rows)
    assert any("final_time differs" in v for v in report["violations"])
    rows = _series(); rows[7]["correctness"]["result"]["checkpoints"]["step_150"]["variables"]["U1"]["sum"] += 1e-3
    code, report = _run(rows)
    assert any("step_150.variables.U1.sum differs" in v for v in report["violations"])
    rows = _series(); rows[8]["correctness"]["result"]["checkpoints"]["step_150"]["variables"]["U1"]["sum"] += 1e-7
    code, report = _run(rows)
    assert code == 0, report["violations"]
    rows = _series(); rows[0]["io_validation"]["expected"]["files_per_checkpoint"] = 1
    code, report = _run(rows)
    assert any("io_validation.expected.files_per_checkpoint" in v for v in report["violations"])
    rows = _series(); rows = rows[:-1]
    code, report = _run(rows)
    assert any("no control run" in v for v in report["violations"])
    rows = _series(); rows[1], rows[2] = rows[2], rows[1]   # N1 r0 then N1 r1 back to back? no: swap makes N2 r0, N2 r1 adjacent
    rows[1]["start_iso"], rows[2]["start_iso"] = rows[2]["start_iso"], rows[1]["start_iso"]
    code, report = _run(rows)
    assert any(v.startswith("order:") for v in report["violations"])
    rows = _series(); rows[4]["correctness"]["pass"] = False
    code, report = _run(rows)
    assert any("correctness check is missing, malformed, or failed" in v for v in report["violations"])


HISTORY = "wrfout_d01_2019-11-27_00_00_00"
WRF_WORK = {"forecast_start": "2019-11-26_23:00:00", "forecast_end": "2019-11-27_00:00:00", "simulated_seconds": 3600,
            "time_step_s": 72, "expected_time_steps": 50, "history_interval_m": 60, "expected_history_frames": 1,
            "frames_per_outfile": 1, "grid_we": 425, "grid_sn": 300, "grid_vert": 50, "ranks": 128, "nodes": 2,
            "physics_suite": "conus", "io_form_restart": 2, "io_form_boundary": 2, "restart_sha256": "r",
            "boundary_sha256": "b", "namelist_normalized_sha256": "n"}
WRF_IO = {"W1": {"write_module": "POSIX", "shared_file": False, "writing_ranks": 1, "collective": False,
                 "data_model": "NETCDF3_64BIT_OFFSET", "io_form_history": 2, "history_files": 1},
          "W2": {"write_module": "MPI-IO", "shared_file": True, "collective": True,
                 "data_model": "NETCDF3_64BIT_OFFSET", "io_form_history": 11, "history_files": 1}}


def _wrf_result():
    return {"success_complete": True, "final_time": "2019-11-27_00:00:00", "completed_steps": 50, "history_frames": 1,
            "history_files": [HISTORY],
            "history": {HISTORY: {"data_model": "NETCDF3_64BIT_OFFSET", "dimensions": {"Time": 1, "west_east": 424},
                                  "global_attributes": {"TITLE": "x"},
                                  "variables": {"T2": {"dtype": "<f4", "sha256": "t2", "sum": 1.0},
                                                "U": {"dtype": "<f4", "sha256": "u", "sum": 2.0}}}}}


def _wrf_row(case, repeat, wall, write_s, control=False, start=0):
    return {"app": "wrf", "attempt": "t", "case": case, "role": "problem" if case == "W1" else "fix", "kind": "natural",
            "repeat": repeat, "jobid": f"{case}{repeat}{'c' if control else ''}", "nodes": 2, "ntasks": 128,
            "nodelist": "cn[1-2]", "start_iso": f"2026-09-24T01:{start:02d}:00-05:00", "wall_s": wall, "rc": 0,
            "nodarshan": control, "app_metric": {"name": "history_write_s", "value": write_s},
            "knobs": {"io_form_history": WRF_IO[case]["io_form_history"]}, "work": dict(WRF_WORK),
            "correctness": {"check": "wrf_success_history_frames_and_fields", "pass": True, "result": _wrf_result()},
            "io_validation": {"check": "control_run_without_darshan" if control else "history_file_darshan_organization",
                              "pass": True, "expected": dict(WRF_IO[case], ranks=128), "observed": {"history_files": [{"name": HISTORY}]},
                              "modules": [] if control else ["POSIX"]},
            "out_files": 1, "out_bytes": 595555580, "scratch_root": "/s", "darshan_logs": [] if control else [f"/d/{case}{repeat}.darshan"],
            "binary_sha256": "b", "script": "/s.slurm", "script_sha256": "s", "iosage_commit": "c"}


def _wrf_series(repeats=11, w1=(53.0, 4.3), w2=(44.0, 1.3)):
    rows, minute = [], 0
    for k in range(repeats):
        for case, (wall, write) in (("W1", w1), ("W2", w2)):
            rows.append(_wrf_row(case, k, wall + 0.1 * k, write + 0.01 * k, start=minute)); minute += 1
    for case, (wall, write) in (("W1", w1), ("W2", w2)):
        rows.append(_wrf_row(case, 0, wall, write, control=True, start=minute)); minute += 1
    return rows


def _run_wrf(rows, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp, "manifest.jsonl")
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = Path(tmp, "audit.json")
        original_argv, original_summary = sys.argv, audit.darshan_summary
        sys.argv = ["audit", "--manifest", str(manifest), "--cases", str(CASES), "--app", "wrf", "--output", str(out), *extra]
        audit.darshan_summary = lambda path: {"written": 0, "modules": ["POSIX"], "partial": [], "nprocs": 128}
        try:
            audit.main()
        except SystemExit as exc:
            code = exc.code
        finally:
            sys.argv, audit.darshan_summary = original_argv, original_summary
        return code, json.loads(out.read_text())


def test_wrf_series_reports_primary_and_secondary_verdicts():
    code, report = _run_wrf(_wrf_series())
    assert code == 0, report["violations"]
    verdict = report["verdicts"]["W1:W2"]
    assert verdict["verdict"] == "faster" and verdict["secondary"]["metric"] == "history_write_s"
    assert verdict["secondary"]["verdict"] == "faster" and verdict["secondary"]["problem_median"] > verdict["secondary"]["fix_median"]
    assert report["classifier_support"]["W1"]["supported"] is False
    rows = _wrf_series(); rows[3]["correctness"]["result"]["history"][HISTORY]["variables"]["U"]["sha256"] = "changed"
    code, report = _run_wrf(rows)
    assert code == 1 and any("variables differs in ['U']" in v for v in report["violations"])
    rows = _wrf_series(); rows[2]["io_validation"]["expected"]["collective"] = True
    code, report = _run_wrf(rows)
    assert any("io_validation.expected.collective" in v for v in report["violations"])
    rows = _wrf_series(); rows[4]["app_metric"] = {"name": "other", "value": 1.0}
    code, report = _run_wrf(rows)
    assert any("secondary metric" in v for v in report["violations"])
    rows = _wrf_series(w1=(44.0, 1.3))
    code, report = _run_wrf(rows)
    assert code == 0 and report["verdicts"]["W1:W2"]["verdict"] == "no_significant_change"


def test_exact_subtree_rule_compares_whole_structures():
    left = {"history": {"f": {"variables": {"T2": {"sha256": "a", "sum": 1.0}, "U": {"sha256": "b"}}}}}
    right = {"history": {"f": {"variables": {"T2": {"sha256": "a", "sum": 1.0}, "U": {"sha256": "b"}}}}}
    rule = {"exact_subtrees": ["history.f.variables"]}
    assert audit.equivalent_results(left, right, rule) == (True, None)
    right["history"]["f"]["variables"]["U"]["sha256"] = "c"
    passed, reason = audit.equivalent_results(left, right, rule)
    assert not passed and "['U']" in reason
    passed, reason = audit.equivalent_results(left, {"history": {}}, rule)
    assert not passed and "cannot compare" in reason
    assert not audit.equivalent_results({"a": {}}, {"a": {}}, {"exact_subtrees": ["a"]})[0]


if __name__ == "__main__":
    test_valid_series_passes_and_reports_the_verdict_as_produced()
    test_protocol_and_evidence_violations_are_reported()
    print("app audit tests pass")
