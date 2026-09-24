import hashlib
from pathlib import Path
from uuid import uuid4

from scripts.apps.audit_app_runs import (
    correctness_passed,
    equivalent_results,
    io_validation_passed,
)
from scripts.apps.null_check import correctness_passed as null_correctness_passed
from scripts.verify_app_smoke_runs import move_job_files
from scripts.verify_smoke_scenario import parse_labels
from scripts.build_label_manifest import apply_manifest_policy, label_string_to_dims


def _raises(call, exception):
    try:
        call()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def test_correctness_requires_named_boolean_success():
    bad_values = [None, {}, {"check": None, "pass": True}, {"check": "", "pass": True},
                  {"check": "mass", "pass": None}, {"check": "mass", "pass": False}]
    for value in bad_values:
        row = {"correctness": value}
        assert not correctness_passed(row)
        assert not null_correctness_passed(row)


def test_io_validation_requires_named_boolean_success():
    assert io_validation_passed({"io_validation": {"check": "per_file_bytes", "pass": True}})
    assert not io_validation_passed({"io_validation": {"check": "per_file_bytes", "pass": None}})
    assert not io_validation_passed({})


def test_numeric_equivalence_rejects_out_of_tolerance_and_nonfinite_values():
    rule = {"fields": {"energy.total": {"absolute": 0.01, "relative": 0.0}}}
    assert equivalent_results({"energy": {"total": 2.0}}, {"energy": {"total": 2.005}}, rule) == (True, None)
    passed, reason = equivalent_results({"energy": {"total": 2.0}}, {"energy": {"total": 2.02}}, rule)
    assert not passed
    assert "energy.total differs" in reason
    passed, reason = equivalent_results({"energy": {"total": 2.0}}, {"energy": {"total": float("nan")}}, rule)
    assert not passed
    assert "nonfinite" in reason


def test_smoke_label_parser_rejects_unknown_or_zero_labels():
    labels = parse_labels("access_pattern=1,interface_choice=1")
    assert labels["access_pattern"] == 1
    assert labels["interface_choice"] == 1
    _raises(lambda: parse_labels("unknown=1"), ValueError)
    _raises(lambda: parse_labels("healthy=0"), ValueError)


def test_audited_scenarios_have_explicit_target_contracts():
    # the generator labels below are the Label lines of the stored SLURM stdout files
    interleaved, validity, _ = apply_manifest_policy(
        "h5bench", "h5b_interleaved_access_n16_r1", label_string_to_dims("healthy=1"))
    assert interleaved["access_pattern"] == 0
    assert validity["valid_access_pattern"] == 1

    independent, validity, note = apply_manifest_policy(
        "h5bench", "h5b_indep_small_n32_r2", label_string_to_dims("interface_choice=1"))
    assert independent["access_granularity"] == 1
    assert independent["interface_choice"] == 1
    assert validity["valid_interface_choice"] == 1
    assert "policy adds access_granularity" in note

    excluded, validity, _ = apply_manifest_policy(
        "h5bench", "h5b_indep_small_single_ost_n32_r2",
        label_string_to_dims("access_granularity=1,interface_choice=1,throughput_utilization=1"))
    assert excluded is None and validity is None

    for scenario, generator_label in (
        ("hacc_posix_shared_single_ost_p500000_n64_r1", "throughput_utilization=1"),
        ("hacc_posix_shared_many_single_ost_p500000_n64_r1",
         "interface_choice=1,throughput_utilization=1"),
    ):
        excluded, validity, _ = apply_manifest_policy(
            "hacc_io", scenario, label_string_to_dims(generator_label))
        assert excluded is None and validity is None


def test_passed_smoke_evidence_is_moved_with_hashes():
    tmp_path = Path(".codex-trash") / f"batch4_smoke_move_test_{uuid4().hex}"
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True)
    stdout = evidence / "job.out"
    stderr = evidence / "job.err"
    darshan = evidence / "job.darshan"
    stdout.write_bytes(b"stdout")
    stderr.write_bytes(b"stderr")
    darshan.write_bytes(b"darshan")
    result = {"jobid": "17", "stdout": str(stdout), "darshan": [{"path": str(darshan)}]}

    records = move_job_files(result, tmp_path / "trash")

    assert len(records) == 3
    assert all(not (evidence / name).exists() for name in ("job.out", "job.err", "job.darshan"))
    assert Path(result["stdout"]).read_bytes() == b"stdout"
    assert Path(result["darshan"][0]["path"]).read_bytes() == b"darshan"
    hashes = {record["sha256"] for record in records}
    assert hashlib.sha256(b"stdout").hexdigest() in hashes
    assert hashlib.sha256(b"darshan").hexdigest() in hashes
