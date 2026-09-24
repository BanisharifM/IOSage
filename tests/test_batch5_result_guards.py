"""Regression checks for result identity and reproduction guards."""

import csv
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from scripts.analyze_dataset import module_aware_io_bytes
from scripts.recompute_iterative_walltime import locate_stage
from scripts.run_e2e_pipeline_analysis import (
    load_correctness,
    load_manifest,
    select_correctness,
    select_one,
)
from scripts.run_tracebench_full_evaluation import require_common_system_results
from scripts.run_wisio_baseline import assign_trace_paths
from scripts.verify_groundedness_atomic import verify_document
from src.artifact_paths import checked_output_dir
from src.llm.iterative_result import load_iterative_results, validate_iterative_result
from src.llm.iterative_optimizer import publish_results


def _raises(call, exception):
    try:
        call()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def _iterative_record(run_id=1, condition="full"):
    return {
        "schema_version": 2,
        "workload": "ior_small_posix",
        "model": "gpt-4o",
        "run_id": run_id,
        "condition": condition,
        "result_id": f"ior_small_posix:gpt-4o:{condition}:{run_id}",
        "final_status": "converged",
        "max_iterations": 3,
        "config": {
            "use_ml": True,
            "use_shap": True,
            "use_kb": True,
            "use_feedback": True,
            "dry_run": False,
        },
        "iterations": [{"executed": True, "speedup": 1.2}],
        "best_speedup": 1.2,
        "total_iterations": 1,
        "total_cost_usd": 0.01,
        "total_tokens": 10,
        "total_execution_time_s": 2.0,
    }


def _scratch(name):
    path = Path(".codex-trash") / f"batch5_{name}_{uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_iterative_loader_rejects_unknown_schema_and_duplicate_identity():
    valid = _iterative_record()
    assert validate_iterative_result(valid) is valid

    old = dict(valid, schema_version=0)
    _raises(lambda: validate_iterative_result(old), ValueError)
    simulated = dict(valid, config=dict(valid["config"], dry_run=True))
    _raises(lambda: validate_iterative_result(simulated), ValueError)

    result_dir = _scratch("iterative_loader")
    result_file = result_dir / "sweep_run.json"
    result_file.write_text(json.dumps([valid, valid]))
    _raises(lambda: load_iterative_results(result_dir), ValueError)

    valid_dir = _scratch("iterative_valid")
    (valid_dir / "sweep_run.json").write_text(json.dumps(valid))
    assert load_iterative_results(valid_dir)[0]["result_id"] == valid["result_id"]


def test_iterative_conditions_have_distinct_ids_and_failed_runs_are_not_published():
    result_dir = _scratch("iterative_conditions")
    full = _iterative_record(condition="full")
    no_kb = _iterative_record(condition="no_kb")
    source = result_dir / "sweep_and_ablation.json"
    publish_results([full, no_kb], source)
    assert {record["condition"] for record in load_iterative_results(result_dir)} == {
        "full", "no_kb"}

    failed = dict(full, final_status="baseline_failed")
    rejected = result_dir / "sweep_rejected.json"
    _raises(lambda: publish_results([failed], rejected), ValueError)
    assert not rejected.exists()


def test_generated_output_path_rejects_frozen_paper():
    active = checked_output_dir("papers/IPDPS_2027/figures")
    assert "IPDPS_2027" in active.parts
    _raises(lambda: checked_output_dir("papers/SC_2026/figures"), ValueError)
    _raises(lambda: checked_output_dir("paper/figures"), ValueError)


def test_module_aware_bytes_handles_mpiio_and_avoids_layer_double_counting():
    frame = pd.DataFrame({
        "POSIX_BYTES_READ": [0, 100],
        "POSIX_BYTES_WRITTEN": [0, 100],
        "MPIIO_BYTES_READ": [300, 100],
        "MPIIO_BYTES_WRITTEN": [200, 100],
        "STDIO_BYTES_READ": [0, 5],
        "STDIO_BYTES_WRITTEN": [0, 5],
    })
    assert module_aware_io_bytes(frame).tolist() == [500.0, 200.0]


def test_walltime_match_rejects_out_of_tolerance_and_ambiguous_jobs():
    with patch("scripts.recompute_iterative_walltime.candidate_jobs",
               return_value=[("1", "one.out")]), patch(
                   "scripts.recompute_iterative_walltime.job_logs",
                   return_value=["one.darshan"]), patch(
                   "scripts.recompute_iterative_walltime.executor_bw",
                   return_value=80.0):
        _raises(lambda: locate_stage("ior", "gpt", "baseline", 100.0), ValueError)

    with patch("scripts.recompute_iterative_walltime.candidate_jobs",
               return_value=[("1", "one.out"), ("2", "two.out")]), patch(
                   "scripts.recompute_iterative_walltime.job_logs",
                   return_value=["trace.darshan"]), patch(
                   "scripts.recompute_iterative_walltime.executor_bw",
                   return_value=100.0):
        _raises(lambda: locate_stage("ior", "gpt", "baseline", 100.0), ValueError)


def test_tracebench_requires_identical_trace_coverage():
    traces = [{"trace_key": "a", "iosage_metrics": {"n_tp": 1},
               "ionavigator_metrics": {"n_tp": 1}, "drishti_metrics": None}]
    _raises(lambda: require_common_system_results(
        traces, ["iosage", "ionavigator", "drishti"]), RuntimeError)
    traces[0]["drishti_metrics"] = {"n_tp": 1}
    aligned = require_common_system_results(
        traces, ["iosage", "ionavigator", "drishti"])
    assert all(len(values) == 1 for values in aligned.values())


def test_wisio_assigns_every_trace_once_using_declared_cardinality():
    logs = _scratch("wisio_assignment") / "logs"
    (logs / "h5bench").mkdir(parents=True)
    (logs / "dlio").mkdir()
    for name in (
            "read_id10-1.darshan", "write_id10-2.darshan"):
        (logs / "h5bench" / name).write_bytes(b"trace")
    for name in ("rank_id20-1.darshan", "rank_id20-2.darshan"):
        (logs / "dlio" / name).write_bytes(b"trace")
    labels = pd.DataFrame({
        "job_id": ["10", "10", "20"],
        "benchmark": ["h5bench", "h5bench", "dlio"],
        "scenario": ["read", "write", "train"],
        "n_darshan_files": [1, 1, 2],
    })
    assignments = assign_trace_paths(labels, logs)
    paths = [path for group in assignments.values() for path in group]
    assert len(assignments) == 3
    assert len(paths) == len(set(paths)) == 4


def test_e2e_manifest_requires_unique_declared_runs():
    scratch = _scratch("e2e_manifest")
    log = scratch / "app_id77-1.darshan"
    log.write_bytes(b"trace")
    manifest = scratch / "runs.tsv"
    fields = ["config_id", "fix_id", "job_id", "executable", "output",
              "darshan_log", "completed_utc"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "config_id": "before", "fix_id": "none", "job_id": "77",
            "executable": "/bin/app", "output": "before.nc4",
            "darshan_log": str(log), "completed_utc": "2026-09-23T00:00:00Z",
        })
    _, rows = load_manifest(manifest)
    before = select_one(rows, "before")
    assert before["job_id"] == "77"
    _raises(lambda: select_one(rows, "after"), ValueError)

    after = dict(before, config_id="after", fix_id="fix", output="after.nc4")
    correctness_path = scratch / "correctness.tsv"
    correctness_fields = [
        "before_config", "after_config", "job_id", "before_output",
        "after_output", "checker", "passed", "completed_utc",
    ]
    with correctness_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=correctness_fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "before_config": "before", "after_config": "after", "job_id": "77",
            "before_output": "before.nc4", "after_output": "after.nc4",
            "checker": "h5diff --quiet", "passed": "true",
            "completed_utc": "2026-09-23T00:00:01Z",
        })
    _, correctness_rows = load_correctness(correctness_path)
    assert select_correctness(correctness_rows, before, after)["passed"] == "true"


def test_absent_recommendations_are_reported_outside_grounding_results():
    entry = {
        "entry_id": "kb2:ior/1/a.darshan",
        "sample_id": "ior/1/a.darshan",
        "job_group": "ior/1",
        "benchmark": "ior",
        "scenario": "small",
        "bottleneck_labels": ["access_granularity"],
        "darshan_signature": {"avg_write_size": 64.0},
        "source_code": {"repository": "repo", "revision": "abc", "path": "io.c"},
        "fixes": [{
            "fix_id": "buffer", "dimension": "access_granularity",
            "description": "buffer writes", "code_before": "write small",
            "code_after": "write large", "api_change": "write size",
            "measurement": {
                "metric": "walltime_speedup", "speedup": 1.25,
                "accepted": True, "correctness_passed": True,
                "source_revision": "abc", "before_job_ids": ["1"],
                "after_job_ids": ["2"], "before_commands": ["before"],
                "after_commands": ["after"], "before_walltime_s": [10.0],
                "after_walltime_s": [8.0], "result_paths": ["result.json"],
                "correctness_result_path": "correctness.json",
            },
        }],
        "shap_top_features": {},
    }
    report = verify_document([
        {"workload": "w", "recommendation": {"parsed": None}}
    ], {"schema_version": 2, "allowed_sample_ids": ["ior/1/a.darshan"],
        "allowed_job_groups": ["ior/1"], "entries": [entry]})
    assert report["checked_results"] == 0
    assert report["absent_recommendations"] == 1
    assert report["results"] == []
