"""Nek5000 evidence contract: field-file reader, checkpoint reassembly, log parsing, the
correctness and I/O validation checks, and the manifest row contract."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps import manifest  # noqa: E402
from scripts.apps.nek5000 import evidence, fld  # noqa: E402

NX = 2
NXYZ = NX ** 3
NEL = 6


def _fields(seed):
    rng = np.random.default_rng(seed)
    return {"X": rng.normal(size=(NEL, 3, NXYZ)), "U": rng.normal(size=(NEL, 3, NXYZ)),
            "P": rng.normal(size=(NEL, 1, NXYZ))}


def _header(nelo, nfileo, fid, istep, time=0.25):
    return {"wdsize": 8, "nx": NX, "ny": NX, "nz": NX, "nelo": nelo, "nelog": NEL, "time": time,
            "istep": istep, "fid": fid, "nfileo": nfileo, "varcode": "XUP"}


def _write_checkpoint(directory, index, istep, data, nfiles):
    """One checkpoint as one shared file or as ``nfiles`` per-group files with a permuted
    element order, mirroring the two Nek5000 output organizations."""
    elements = np.arange(1, NEL + 1)
    if nfiles == 1:
        order = np.array([3, 1, 5, 2, 6, 4]) - 1
        fld.write_file(directory / f"turbChannel0.f{index:05d}", _header(NEL, 1, 0, istep), elements[order],
                       {k: v[order] for k, v in data.items()})
        return
    groups = np.array_split(np.array([6, 2, 4, 1, 5, 3]) - 1, nfiles)
    for fid, group in enumerate(groups):
        fld.write_file(directory / f"turbChannel{fid:02d}.f{index:05d}", _header(len(group), nfiles, fid, istep),
                       elements[group], {k: v[group] for k, v in data.items()})


def _log_text(final_step=100, successful=True):
    lines = [f"Step    {s}, t= {s * 0.01:.7E}, DT= 1.0000000E-02, C=  2.500 1.0E+00 1.0E-01"
             for s in (1, 50, final_step)]
    lines.append(f"        {final_step}  Volflow X              1.0000E+00   3.7109E-03   3.7575E-04   6.2828E+00   6.2832E+00")
    if successful:
        lines.append("run successful: dying ...")
    lines.append("total elapsed time             :   3.31334E+01 sec")
    return "\n".join(lines) + "\n"


def test_reader_round_trip_and_reassembly_is_layout_independent():
    data = _fields(1)
    with tempfile.TemporaryDirectory() as tmp:
        one = Path(tmp, "one"); many = Path(tmp, "many")
        one.mkdir(); many.mkdir()
        _write_checkpoint(one, 1, 50, data, 1)
        _write_checkpoint(many, 1, 50, data, 3)
        s_one = fld.summarize(fld.reassemble(sorted(one.iterdir())))
        s_many = fld.summarize(fld.reassemble(sorted(many.iterdir())))
    assert s_one["files"] == 1 and s_many["files"] == 3 and s_many["per_file_elements"] == [2, 2, 2]
    for key in ("istep", "time", "nelog", "varcode", "nxyz"):
        assert s_one[key] == s_many[key]
    assert s_one["variables"] == s_many["variables"], "same data through 1 and 3 files must summarize identically"
    assert s_one["variables"]["U2"]["sha256"] != s_one["variables"]["U1"]["sha256"]
    # the canonical hash is the data in global element order
    expected = np.ascontiguousarray(data["P"][:, 0, :]).tobytes()
    import hashlib
    assert s_one["variables"]["P0"]["sha256"] == hashlib.sha256(expected).hexdigest()


def test_reassembly_rejects_missing_or_inconsistent_files():
    data = _fields(2)
    with tempfile.TemporaryDirectory() as tmp:
        many = Path(tmp)
        _write_checkpoint(many, 1, 50, data, 3)
        files = sorted(many.iterdir())
        try:
            fld.reassemble(files[:2])
        except fld.FldFormatError as exc:
            assert "3 files" in str(exc) or "header says" in str(exc)
        else:
            raise AssertionError("missing file accepted")
        other = _header(2, 3, 0, 51)
        fld.write_file(files[0], other, [6, 2], {k: v[[5, 1]] for k, v in data.items()})
        try:
            fld.reassemble(files)
        except fld.FldFormatError as exc:
            assert "istep" in str(exc)
        else:
            raise AssertionError("mixed checkpoint steps accepted")


def test_log_parsing_reads_final_step_volflow_and_success():
    log = evidence.parse_log(_log_text())
    assert log["final_step"] == 100 and log["run_successful"] and log["volflow_step"] == 100
    assert abs(log["final_time"] - 1.0) < 1e-9 and log["final_courant"] == 2.5
    assert log["volflow"] == [1.0, 3.7109e-03, 3.7575e-04, 6.2828, 6.2832] and log["total_elapsed_s"] == 33.1334
    assert not evidence.parse_log(_log_text(successful=False))["run_successful"]


def _darshan_dict(files, module, shared, nprocs=4):
    records = {}
    for rank, path in enumerate(files):
        size = path.stat().st_size
        entry = {"bytes_written": size, "writes": 3, "ranks": [] if shared else [rank], "shared": shared,
                 "coll_writes": 3 if module == "MPI-IO" else 0}
        records[str(path)] = {module: entry}
        if module == "MPI-IO":
            records[str(path)]["POSIX"] = dict(entry, coll_writes=0)
    return {"files": records, "modules": ["POSIX", module, "STDIO"], "partial": [], "nprocs": nprocs}


def test_correctness_and_io_validation_on_both_organizations():
    data = [_fields(10 + i) for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        for nfiles, module in ((1, "MPI-IO"), (3, "STDIO")):
            scratch = Path(tmp, f"n{nfiles}"); scratch.mkdir()
            for index, istep in enumerate((0, 50, 100), start=1):
                _write_checkpoint(scratch, index, istep, data[index - 1], nfiles)
            groups = evidence.checkpoint_groups(scratch)
            assert list(groups) == [1, 2, 3] and all(len(v) == nfiles for v in groups.values())
            correctness = evidence.build_correctness(evidence.parse_log(_log_text()), groups, 100, 50)
            assert correctness["pass"], correctness["problems"]
            assert list(correctness["result"]["checkpoints"]) == ["step_0", "step_50", "step_100"]
            assert correctness["result"]["checkpoints"]["step_50"]["variables"]["U_minmax0"]["count"] == NEL * 6
            files = [p for paths in groups.values() for p in paths]
            io = evidence.build_io_validation(groups, nfiles, _darshan_dict(files, module, nfiles == 1), 4)
            assert io["pass"], io["problems"]
            assert io["expected"]["write_module"] == module and io["observed"]["files_per_checkpoint"] == [nfiles] * 3
            # wrong organization or short Darshan bytes must fail
            wrong = evidence.build_io_validation(groups, 1 if nfiles == 3 else 3, _darshan_dict(files, module, nfiles == 1), 4)
            assert not wrong["pass"]
            short = _darshan_dict(files, module, nfiles == 1)
            next(iter(short["files"].values()))[module]["bytes_written"] -= 1
            assert not evidence.build_io_validation(groups, nfiles, short, 4)["pass"]
            control = evidence.build_io_validation(groups, nfiles, None, 4)
            assert control["pass"] and control["check"] == "control_run_without_darshan"
        # a missing checkpoint or a failed log fails correctness
        scratch = Path(tmp, "n1")
        groups = evidence.checkpoint_groups(scratch)
        assert not evidence.build_correctness(evidence.parse_log(_log_text(successful=False)), groups, 100, 50)["pass"]
        assert not evidence.build_correctness(evidence.parse_log(_log_text()), groups, 150, 50)["pass"]


def _row():
    return {"app": "nek5000", "attempt": "a", "case": "N1", "role": "fix", "kind": "constructed", "repeat": 0,
            "jobid": "1", "nodes": 1, "ntasks": 32, "nodelist": "cn1", "start_iso": "t", "wall_s": 3.5, "rc": 0,
            "nodarshan": False, "app_metric": {"name": "total_elapsed_s", "value": 3.0},
            "knobs": {"writeNFiles": 1}, "work": {"steps": 200, "precision": "double"},
            "correctness": {"check": "c", "pass": True, "result": {"final_step": 200}},
            "io_validation": {"check": "io", "pass": True, "expected": {"a": 1}, "observed": {"b": 2}, "modules": ["POSIX"]},
            "out_files": 5, "out_bytes": 10, "scratch_root": "/s", "darshan_logs": ["/d/log.darshan"],
            "binary_sha256": "x", "script": "/s.slurm", "script_sha256": "y", "iosage_commit": "z"}


def test_manifest_contract_accepts_typed_rows_and_rejects_untyped_ones():
    assert manifest.validate_row(_row()) == []
    bad = _row(); bad["correctness"]["pass"] = None
    assert any("correctness.pass" in p for p in manifest.validate_row(bad))
    bad = _row(); bad["io_validation"].pop("expected")
    assert any("io_validation.expected" in p for p in manifest.validate_row(bad))
    bad = _row(); bad["work"] = {}
    assert "work is empty" in manifest.validate_row(bad)
    bad = _row(); bad["nodarshan"] = True
    assert "control run lists Darshan logs" in manifest.validate_row(bad)
    bad = _row(); bad["wall_s"] = "3.5"
    assert "wall_s is not a number" in manifest.validate_row(bad)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "manifest.jsonl")
        manifest.append_row(path, _row())
        assert json.loads(path.read_text())["case"] == "N1"
        try:
            manifest.append_row(path, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid row was appended")
        assert len(manifest.read_rows(path)) == 1


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    print("nek5000 evidence tests pass")
