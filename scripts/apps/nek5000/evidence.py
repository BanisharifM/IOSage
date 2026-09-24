"""Build the typed manifest row of one Nek5000 turbChannel run (application study, task 01).

Runs inside the job after ``nek5000`` has finished, while the scratch directory still holds
the field files. It reads the log, every checkpoint file and the job's Darshan log and writes:

* ``work``: steps, checkpoint interval, expected checkpoint count, precision, ranks, nodes,
  mesh checksum (``.re2`` + ``.ma2``) and input checksum (``.par`` without the ``writeNFiles``
  line, ``.usr``, ``SIZE``), so that a problem case and its fix carry identical work;
* ``correctness``: ``nek5000_log_and_checkpoint_fields``: the run printed ``run successful``,
  reached the requested step, wrote every expected checkpoint, each checkpoint reassembles
  all elements exactly once, and every field value is finite. ``result`` holds the final
  step, time, time step and Courant number from the log, the last ``Volflow X`` line, and
  under ``checkpoints.step_<istep>`` the header time and step and per variable component a
  canonical SHA-256, sum, minimum and maximum (``fld.summarize``). The preregistered rule
  (``configs/app_cases.yaml``) compares these with tolerances: two runs of the same case
  already differ at the 1e-12 level from step 50 on, so hash equality is not the criterion;
* ``io_validation``: ``per_file_darshan_bytes_and_organization``: the number of files per
  checkpoint equals ``writeNFiles``; every field file has a Darshan record whose bytes
  written in the expected module (MPI-IO for one shared file, STDIO for per-rank files)
  equal its size on disk; every other module with a record for that file reports the same
  bytes; a shared file is a shared record written collectively, a per-rank file is written
  by exactly one rank; no module has the partial flag. Control runs (``NODARSHAN``) record
  the file organization from disk only, under ``control_run_without_darshan``.

Exit code 0 when both checks pass, 3 otherwise; 2 on a usage or parsing failure.
"""
import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps import manifest  # noqa: E402
from scripts.apps.nek5000 import fld  # noqa: E402

logger = logging.getLogger("nek5000_evidence")

STEP_LINE = re.compile(r"^\s*Step\s+(\d+), t=\s*([-+\dE.]+), DT=\s*([-+\dE.]+), C=\s*([-+\dE.]+)")
VOLFLOW_LINE = re.compile(r"^\s*(\d+)\s+Volflow X\s+(.+?)\s*$")
ELAPSED_LINE = re.compile(r"total elapsed time\s*:\s*([-+\dE.]+)")
FIELD_FILE = re.compile(r"^turbChannel(\d+)\.f(\d{5})$")
WRITE_COUNTERS = {"POSIX": "POSIX_BYTES_WRITTEN", "MPI-IO": "MPIIO_BYTES_WRITTEN", "STDIO": "STDIO_BYTES_WRITTEN"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(text):
    """Final step line, last Volflow line, success flag and Nek's own elapsed time."""
    steps = [m for m in (STEP_LINE.match(line) for line in text.splitlines()) if m]
    volflows = [m for m in (VOLFLOW_LINE.match(line) for line in text.splitlines()) if m]
    elapsed = ELAPSED_LINE.search(text)
    out = {"run_successful": "run successful" in text,
           "final_step": int(steps[-1].group(1)) if steps else None,
           "final_time": float(steps[-1].group(2)) if steps else None,
           "final_dt": float(steps[-1].group(3)) if steps else None,
           "final_courant": float(steps[-1].group(4)) if steps else None,
           "volflow_step": int(volflows[-1].group(1)) if volflows else None,
           "volflow": [float(v) for v in volflows[-1].group(2).split()] if volflows else None,
           "total_elapsed_s": float(elapsed.group(1)) if elapsed else None}
    return out


def checkpoint_groups(scratch):
    """{checkpoint index: [paths]} of the field files in the scratch directory."""
    groups = defaultdict(list)
    for path in Path(scratch).iterdir():
        match = FIELD_FILE.match(path.name)
        if match and path.is_file():
            groups[int(match.group(2))].append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def darshan_file_records(log_path):
    """Per file name: {module: {bytes_written, writes, ranks, shared, coll_writes}} plus module
    names and partial flags of one Darshan log."""
    import darshan

    report = darshan.DarshanReport(str(log_path), read_all=True)
    names = report.name_records
    per_file = defaultdict(dict)
    for module, column in WRITE_COUNTERS.items():
        if module not in report.records:
            continue
        counter_names = report.counters[module]["counters"]
        tag = module.replace("-", "")
        i_bytes = counter_names.index(column)
        i_writes = counter_names.index(f"{tag}_WRITES" if module != "MPI-IO" else "MPIIO_INDEP_WRITES")
        i_coll = counter_names.index("MPIIO_COLL_WRITES") if module == "MPI-IO" else None
        for rec in report.records[module]:
            name = names.get(rec["id"], "")
            counters = rec["counters"]
            entry = per_file[name].setdefault(module, {"bytes_written": 0, "writes": 0, "ranks": set(),
                                                       "shared": False, "coll_writes": 0})
            entry["bytes_written"] += int(counters[i_bytes])
            entry["writes"] += int(counters[i_writes])
            if i_coll is not None:
                entry["coll_writes"] += int(counters[i_coll])
            rank = int(rec["rank"])
            if rank < 0:
                entry["shared"] = True
            else:
                entry["ranks"].add(rank)
    for entries in per_file.values():
        for entry in entries.values():
            entry["ranks"] = sorted(entry["ranks"])
    partial = sorted(m for m, meta in report.modules.items() if meta.get("partial_flag"))
    return {"files": dict(per_file), "modules": sorted(report.modules), "partial": partial,
            "nprocs": int(report.metadata["job"]["nprocs"])}


def build_correctness(log, groups, nsteps, wint):
    expected_steps = list(range(0, nsteps + 1, wint))
    result = {k: log[k] for k in ("run_successful", "final_step", "final_time", "final_dt", "final_courant",
                                  "volflow_step", "volflow")}
    result["checkpoints"] = {}
    problems = []
    if not log["run_successful"]:
        problems.append("log has no 'run successful' line")
    if log["final_step"] != nsteps:
        problems.append(f"final step {log['final_step']} is not {nsteps}")
    if log["volflow_step"] != nsteps:
        problems.append(f"last Volflow line is for step {log['volflow_step']}")
    seen_steps = []
    for index, paths in groups.items():
        try:
            summary = fld.summarize(fld.reassemble(paths))
        except (fld.FldFormatError, OSError, ValueError) as exc:
            problems.append(f"checkpoint {index}: {exc}")
            continue
        summary["index"] = index
        result["checkpoints"][f"step_{summary['istep']}"] = summary
        seen_steps.append(summary["istep"])
    if seen_steps != expected_steps:
        problems.append(f"checkpoint steps {seen_steps} are not {expected_steps}")
    passed = not problems
    return {"check": "nek5000_log_and_checkpoint_fields", "pass": passed, "result": result,
            "problems": problems}


def build_io_validation(groups, nfiles, darshan, nprocs):
    expected = {"files_per_checkpoint": nfiles, "write_module": "MPI-IO" if nfiles == 1 else "STDIO",
                "shared_file": nfiles == 1, "ranks": nprocs}
    observed = {"files_per_checkpoint": [len(paths) for paths in groups.values()],
                "field_files": sum(len(paths) for paths in groups.values()),
                "field_bytes": sum(p.stat().st_size for paths in groups.values() for p in paths),
                "files": []}
    problems = []
    if not groups:
        problems.append("no field files on disk")
    for index, paths in groups.items():
        if len(paths) != nfiles:
            problems.append(f"checkpoint {index}: {len(paths)} files, expected {nfiles}")
    if darshan is None:
        return {"check": "control_run_without_darshan", "pass": not problems, "expected": expected,
                "observed": observed, "modules": [], "problems": problems}
    observed["modules"] = darshan["modules"]
    observed["partial_modules"] = darshan["partial"]
    if darshan["partial"]:
        problems.append(f"partial Darshan modules {darshan['partial']}")
    for index, paths in groups.items():
        for path in paths:
            size = path.stat().st_size
            records = darshan["files"].get(str(path), {})
            row = {"name": path.name, "size": size,
                   "darshan": {m: {"bytes_written": r["bytes_written"], "writes": r["writes"],
                                   "ranks": len(r["ranks"]), "shared": r["shared"], "coll_writes": r["coll_writes"]}
                               for m, r in records.items()}}
            observed["files"].append(row)
            main = records.get(expected["write_module"])
            if main is None:
                problems.append(f"{path.name}: no {expected['write_module']} record")
                continue
            if main["bytes_written"] != size:
                problems.append(f"{path.name}: {expected['write_module']} wrote {main['bytes_written']} B, size {size} B")
            for module, rec in records.items():
                if rec["bytes_written"] not in (0, size):
                    problems.append(f"{path.name}: {module} record reports {rec['bytes_written']} B, size {size} B")
            if expected["shared_file"]:
                if not main["shared"] or main["coll_writes"] <= 0:
                    problems.append(f"{path.name}: expected a shared record with collective writes")
            elif main["shared"] or len(main["ranks"]) != 1:
                problems.append(f"{path.name}: expected exactly one writing rank, got shared={main['shared']} "
                                f"ranks={main['ranks']}")
    return {"check": "per_file_darshan_bytes_and_organization", "pass": not problems, "expected": expected,
            "observed": observed, "modules": darshan["modules"], "problems": problems}


def input_checksum(scratch, case_dir):
    par_lines = [line for line in (Path(scratch) / "turbChannel.par").read_text().splitlines()
                 if not line.startswith("writeNFiles")]
    digest = hashlib.sha256("\n".join(par_lines).encode())
    for name in ("turbChannel.usr", "SIZE"):
        digest.update(Path(case_dir, name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--case-dir", required=True, help="directory with the mesh, .usr and SIZE")
    parser.add_argument("--logfile", required=True)
    parser.add_argument("--nsteps", type=int, required=True)
    parser.add_argument("--write-interval", type=int, required=True)
    parser.add_argument("--nfiles", type=int, required=True)
    parser.add_argument("--double-precision", required=True, choices=["yes", "no"])
    parser.add_argument("--darshan-logpath", required=True)
    parser.add_argument("--nodarshan", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-json", required=True, help="full evidence written here (kept with the logs)")
    for name in ("app", "attempt", "case", "role", "kind", "jobid", "nodelist", "start-iso", "script",
                 "binary", "darshan-config", "iosage-commit", "scratch-root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--ntasks", type=int, required=True)
    parser.add_argument("--wall-s", type=float, required=True)
    parser.add_argument("--rc", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scratch = Path(args.scratch)
    log = parse_log(Path(args.logfile).read_text(errors="replace"))
    groups = checkpoint_groups(scratch)
    darshan_logs = sorted(str(p) for p in Path(args.darshan_logpath).glob(f"*id{args.jobid}-*"))
    darshan = None
    if not args.nodarshan:
        if len(darshan_logs) != 1:
            logger.error("expected one Darshan log for job %s, found %s", args.jobid, darshan_logs)
            sys.exit(2)
        darshan = darshan_file_records(darshan_logs[0])
        if darshan["nprocs"] != args.ntasks:
            logger.error("Darshan nprocs %d differs from ntasks %d", darshan["nprocs"], args.ntasks)
            sys.exit(2)
    correctness = build_correctness(log, groups, args.nsteps, args.write_interval)
    io_validation = build_io_validation(groups, args.nfiles, darshan, args.ntasks)
    stripe = subprocess.run(["lfs", "getstripe", "-d", str(scratch)], capture_output=True, text=True, check=False)
    row = {
        "app": args.app, "attempt": args.attempt, "case": args.case, "role": args.role, "kind": args.kind,
        "repeat": args.repeat, "jobid": args.jobid, "nodes": args.nodes, "ntasks": args.ntasks,
        "nodelist": args.nodelist, "start_iso": args.start_iso, "wall_s": args.wall_s, "rc": args.rc,
        "nodarshan": args.nodarshan,
        "app_metric": {"name": "total_elapsed_s", "value": log["total_elapsed_s"]},
        "knobs": {"numSteps": args.nsteps, "writeInterval": args.write_interval, "writeNFiles": args.nfiles,
                  "writeDoublePrecision": args.double_precision},
        "work": {"steps": args.nsteps, "checkpoint_interval": args.write_interval,
                 "expected_checkpoints": args.nsteps // args.write_interval + 1,
                 "precision": "double" if args.double_precision == "yes" else "single",
                 "ranks": args.ntasks, "nodes": args.nodes,
                 "mesh_sha256": hashlib.sha256((sha256_file(Path(args.case_dir, "turbChannel.re2"))
                                                + sha256_file(Path(args.case_dir, "turbChannel.ma2"))).encode()).hexdigest(),
                 "input_sha256": input_checksum(scratch, args.case_dir)},
        "correctness": {k: correctness[k] for k in ("check", "pass", "result")},
        "io_validation": {k: io_validation[k] for k in ("check", "pass", "expected", "observed", "modules")},
        "out_files": io_validation["observed"]["field_files"], "out_bytes": io_validation["observed"]["field_bytes"],
        "stripe": " ".join(stripe.stdout.split()) if stripe.returncode == 0 else None,
        "scratch_root": args.scratch_root, "darshan_logs": [] if args.nodarshan else darshan_logs,
        "darshan_conf_sha256": sha256_file(args.darshan_config) if Path(args.darshan_config).is_file() else None,
        "binary_sha256": sha256_file(args.binary), "script": args.script, "script_sha256": sha256_file(args.script),
        "iosage_commit": args.iosage_commit,
    }
    evidence = {"row": row, "correctness_problems": correctness["problems"],
                "io_problems": io_validation["problems"], "log": log}
    Path(args.evidence_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.evidence_json).write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str))
    for problem in correctness["problems"] + io_validation["problems"]:
        logger.warning("CHECK FAILED: %s", problem)
    manifest.append_row(args.manifest, row)
    logger.info("manifest row appended: %s (correctness %s, io_validation %s)", args.manifest,
                correctness["pass"], io_validation["pass"])
    sys.exit(0 if correctness["pass"] and io_validation["pass"] else 3)


if __name__ == "__main__":
    main()
