"""Build the typed manifest row of one WRF CONUS 12 km run (application study, task 02).

Runs inside the job after ``wrf`` has finished, while the scratch directory still holds the
history file. It reads ``rsl.out.0000``, the namelist, every history file (through netCDF4)
and the job's Darshan log, and writes:

* ``work``: forecast start and end, simulated seconds, completed time steps, time step,
  history interval, expected history frames, grid dimensions, ranks, nodes, physics suite,
  restart checksum, boundary-file checksum and the checksum of the namelist without its
  ``io_form_history`` line, so that a problem run and its fix run carry identical work;
* ``correctness``: ``wrf_success_history_frames_and_fields``: ``SUCCESS COMPLETE WRF`` was
  printed, the final simulated time equals the registered end time, the completed step count
  matches the forecast length, and every expected history frame exists and opens.
  ``result`` holds the final time, the step count and, per history file, the data model, the
  dimensions, the global attributes and per variable the data type, dimensions, shape, a
  SHA-256 of the raw values and, for numeric variables, count, minimum, maximum and sum;
* ``io_validation``: ``history_file_darshan_organization``: the named history file exists,
  its data model is the registered one, the module registered for the case wrote it
  (``POSIX`` by one rank for serial NetCDF, ``MPI-IO`` shared with collective writes for
  PnetCDF), the bytes written in that module are at least the file size, no other write
  layer wrote it, and no module has the partial flag. The observed record keeps every module's
  bytes, calls, collective calls and ranks for the history file and the totals of the restart
  and boundary reads. Control runs (``NODARSHAN``) record the file organization from disk only;
* metrics: ``app_metric`` = ``history_write_s``, the sum of WRF's ``Timing for Writing wrfout``
  seconds (the total wall time of the launch is ``wall_s``, the primary metric);
* provenance: case, role, kind, repeat, job id, commit, script hash, binary hash, WRF source
  revision and local patch hash, namelist hash, Darshan config hash, node list, Darshan log.

Exit code 0 when both checks pass, 3 otherwise; 2 on a usage or parsing failure.
"""
import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps import manifest  # noqa: E402

logger = logging.getLogger("wrf_evidence")

STEP_LINE = re.compile(r"^Timing for main: time (\S+) on domain\s+(\d+):\s+([\d.]+) elapsed seconds")
WRITE_LINE = re.compile(r"^Timing for Writing (\S+) for domain\s+(\d+):\s+([\d.]+) elapsed seconds")
NAMELIST_LINE = re.compile(r"^\s*([A-Za-z_0-9]+)\s*=\s*(.+?)\s*,?\s*$")
WRITE_COUNTERS = {"POSIX": "POSIX_BYTES_WRITTEN", "MPI-IO": "MPIIO_BYTES_WRITTEN", "STDIO": "STDIO_BYTES_WRITTEN"}
READ_COUNTERS = {"POSIX": "POSIX_BYTES_READ", "MPI-IO": "MPIIO_BYTES_READ", "STDIO": "STDIO_BYTES_READ"}
EXPECTED_BY_IO_FORM = {
    # io_netcdf in classic mode creates the file with NF_64BIT_OFFSET (external/io_netcdf/wrf_io.F90,
    # ext_ncd_open_for_write_begin), the same data model PnetCDF uses; without use_netcdf_classic
    # it creates a NetCDF-4 (HDF5) file with deflate level 2, the shipped default
    2: {"write_module": "POSIX", "shared_file": False, "writing_ranks": 1, "collective": False,
        "data_model": "NETCDF3_64BIT_OFFSET"},
    11: {"write_module": "MPI-IO", "shared_file": True, "writing_ranks": None, "collective": True,
         "data_model": "NETCDF3_64BIT_OFFSET"},
}
SHIPPED_SERIAL_DATA_MODEL = "NETCDF4"
WORK_KEYS = ("run_days", "run_hours", "run_minutes", "run_seconds", "history_interval_m", "frames_per_outfile",
             "time_step", "e_we", "e_sn", "e_vert", "max_dom", "restart", "io_form_restart", "io_form_boundary",
             "io_form_input", "nocolons", "use_netcdf_classic", "physics_suite", "nio_tasks_per_group", "nio_groups")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_namelist(text):
    """{key: value string} of every ``key = value`` line (first occurrence wins)."""
    values = {}
    for line in text.splitlines():
        match = NAMELIST_LINE.match(line)
        if match and match.group(1) not in values:
            values[match.group(1)] = match.group(2).strip().rstrip(",").strip()
    return values


def namelist_int(values, key):
    return int(values[key].split(",")[0])


def namelist_str(values, key):
    return values[key].split(",")[0].strip().strip("'\"")


def date_string(values, prefix):
    return "{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}".format(*(namelist_int(values, f"{prefix}_{k}")
                                                                for k in ("year", "month", "day", "hour", "minute", "second")))


def parse_rsl(text):
    steps = [STEP_LINE.match(line) for line in text.splitlines()]
    steps = [m for m in steps if m]
    writes = [m for m in (WRITE_LINE.match(line) for line in text.splitlines()) if m]
    return {"success": "SUCCESS COMPLETE WRF" in text,
            "completed_steps": len(steps),
            "final_time": steps[-1].group(1) if steps else None,
            "step_seconds": [float(m.group(3)) for m in steps],
            "history_writes": [{"file": m.group(1), "seconds": float(m.group(3))} for m in writes],
            "history_write_s": round(sum(float(m.group(3)) for m in writes), 6),
            "tables_computed": [line.strip() for line in text.splitlines() if "computing" in line and "table" in line]}


def history_file_names(values):
    """Expected history file names for one output frame per run (frames_per_outfile 1)."""
    minutes = (namelist_int(values, "run_days") * 1440 + namelist_int(values, "run_hours") * 60
               + namelist_int(values, "run_minutes") + namelist_int(values, "run_seconds") // 60)
    interval = namelist_int(values, "history_interval_m")
    frames = minutes // interval
    end = date_string(values, "end")
    nocolons = namelist_str(values, "nocolons").lower().strip(".") in ("true", "t")
    names = []
    # WRF names a history file after its first frame. The history alarm rings at the first time
    # step at or after each multiple of the interval, so with a 72 s step and a 10-minute
    # interval the frames fall at 23:10:48, 23:20:24, 23:30:00, ... (smoke job 22351338); with
    # frames_per_outfile 1 every frame is its own file.
    start = date_string(values, "start")
    import datetime
    import math
    dt = namelist_int(values, "time_step")
    t0 = datetime.datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    for k in range(1, frames + 1):
        steps = math.ceil(k * interval * 60 / dt)
        stamp = (t0 + datetime.timedelta(seconds=steps * dt)).strftime("%Y-%m-%d_%H:%M:%S")
        names.append("wrfout_d01_" + (stamp.replace(":", "_") if nocolons else stamp))
    return names, frames, end, minutes * 60


def summarize_history(path):
    """Data model, dimensions, global attributes and per-variable summaries of one file."""
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_maskandscale(False)
        out = {"data_model": ds.data_model,
               "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
               "global_attributes": {name: str(ds.getncattr(name)) for name in ds.ncattrs()},
               "variables": {}}
        for name, var in ds.variables.items():
            values = var[:]
            array = np.ascontiguousarray(np.asarray(values))
            entry = {"dtype": array.dtype.str, "dimensions": list(var.dimensions), "shape": list(array.shape),
                     "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            if array.dtype.kind in "fiu" and array.size:
                as_float = array.astype("<f8", copy=False)
                if not np.isfinite(as_float).all():
                    entry["nonfinite"] = int((~np.isfinite(as_float)).sum())
                entry.update({"count": int(array.size), "min": float(as_float.min()), "max": float(as_float.max()),
                              "sum": math.fsum(as_float.ravel().tolist())})
            out["variables"][name] = entry
    return out


def darshan_records(log_path):
    """Per file name: {module: {bytes_written, bytes_read, writes, reads, coll_writes, ranks, shared}},
    module names, partial flags and nprocs."""
    import darshan

    report = darshan.DarshanReport(str(log_path), read_all=True)
    names = report.name_records
    per_file = defaultdict(dict)
    for module in WRITE_COUNTERS:
        if module not in report.records:
            continue
        counter_names = report.counters[module]["counters"]
        idx = {name: i for i, name in enumerate(counter_names)}
        tag = module.replace("-", "")
        for rec in report.records[module]:
            name = names.get(rec["id"], "")
            c = rec["counters"]
            entry = per_file[name].setdefault(module, {"bytes_written": 0, "bytes_read": 0, "writes": 0, "reads": 0,
                                                       "coll_writes": 0, "coll_reads": 0, "ranks": set(), "shared": False})
            entry["bytes_written"] += int(c[idx[WRITE_COUNTERS[module]]])
            entry["bytes_read"] += int(c[idx[READ_COUNTERS[module]]])
            if module == "MPI-IO":
                entry["writes"] += int(c[idx["MPIIO_INDEP_WRITES"]]) + int(c[idx["MPIIO_COLL_WRITES"]])
                entry["reads"] += int(c[idx["MPIIO_INDEP_READS"]]) + int(c[idx["MPIIO_COLL_READS"]])
                entry["coll_writes"] += int(c[idx["MPIIO_COLL_WRITES"]])
                entry["coll_reads"] += int(c[idx["MPIIO_COLL_READS"]])
            else:
                entry["writes"] += int(c[idx[f"{tag}_WRITES"]])
                entry["reads"] += int(c[idx[f"{tag}_READS"]])
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


def build_correctness(rsl, values, scratch):
    names, frames, end, simulated_s = history_file_names(values)
    expected_steps = simulated_s // namelist_int(values, "time_step")
    result = {"success_complete": rsl["success"], "final_time": rsl["final_time"],
              "completed_steps": rsl["completed_steps"], "history_frames": frames,
              "history_files": names, "history": {}}
    problems = []
    if not rsl["success"]:
        problems.append("rsl.out.0000 has no 'SUCCESS COMPLETE WRF'")
    if rsl["final_time"] != end:
        problems.append(f"final simulated time {rsl['final_time']} is not {end}")
    if rsl["completed_steps"] != expected_steps:
        problems.append(f"{rsl['completed_steps']} completed steps, expected {expected_steps}")
    for name in names:
        path = Path(scratch, name)
        if not path.is_file():
            problems.append(f"history file {name} missing")
            continue
        try:
            result["history"][name] = summarize_history(path)
        except Exception as exc:  # reported, never hidden
            problems.append(f"{name}: cannot read: {exc!r}")
    return {"check": "wrf_success_history_frames_and_fields", "pass": not problems, "result": result,
            "problems": problems}


def build_io_validation(names, io_form, scratch, darshan, nprocs, history_models, netcdf_classic=True):
    expected = dict(EXPECTED_BY_IO_FORM[io_form], ranks=nprocs, io_form_history=io_form, history_files=len(names),
                    use_netcdf_classic=netcdf_classic)
    if io_form == 2 and not netcdf_classic:
        expected["data_model"] = SHIPPED_SERIAL_DATA_MODEL
    observed = {"history_files": [], "restart_read": {}, "boundary_read": {}}
    problems = []
    for name in names:
        path = Path(scratch, name)
        if not path.is_file():
            problems.append(f"history file {name} missing")
            continue
        size = path.stat().st_size
        row = {"name": name, "size": size, "data_model": history_models.get(name)}
        if row["data_model"] != expected["data_model"]:
            problems.append(f"{name}: data model {row['data_model']} is not {expected['data_model']}")
        if darshan is not None:
            records = darshan["files"].get(str(path), {})
            row["darshan"] = {m: {k: v for k, v in r.items() if k != "ranks"} | {"ranks": len(r["ranks"])}
                              for m, r in records.items()}
            main = records.get(expected["write_module"])
            if main is None:
                problems.append(f"{name}: no {expected['write_module']} record")
            else:
                if main["bytes_written"] < size:
                    problems.append(f"{name}: {expected['write_module']} wrote {main['bytes_written']} B, size {size} B")
                if expected["shared_file"]:
                    if not main["shared"] or main["coll_writes"] <= 0:
                        problems.append(f"{name}: expected a shared record with collective writes")
                else:
                    if main["shared"] or len(main["ranks"]) != expected["writing_ranks"]:
                        problems.append(f"{name}: expected {expected['writing_ranks']} writing rank, got "
                                        f"shared={main['shared']} ranks={main['ranks']}")
                    if "MPI-IO" in records and records["MPI-IO"]["bytes_written"] > 0:
                        problems.append(f"{name}: MPI-IO wrote the serial NetCDF file")
            if "STDIO" in records and records["STDIO"]["bytes_written"] > 0:
                problems.append(f"{name}: STDIO wrote to the history file")
        observed["history_files"].append(row)
    if darshan is not None:
        observed["modules"] = darshan["modules"]
        observed["partial_modules"] = darshan["partial"]
        if darshan["partial"]:
            problems.append(f"partial Darshan modules {darshan['partial']}")
        for key, pattern in (("restart_read", "wrfrst_d01_"), ("boundary_read", "wrfbdy_d01")):
            for fname, records in darshan["files"].items():
                if Path(fname).name.startswith(pattern):
                    observed[key] = {m: {"bytes_read": r["bytes_read"], "reads": r["reads"], "ranks": len(r["ranks"]),
                                         "shared": r["shared"]} for m, r in records.items()}
        return {"check": "history_file_darshan_organization", "pass": not problems, "expected": expected,
                "observed": observed, "modules": darshan["modules"], "problems": problems}
    return {"check": "control_run_without_darshan", "pass": not problems, "expected": expected,
            "observed": observed, "modules": [], "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--base-dir", required=True, help="directory with namelist.input.base and the input links")
    parser.add_argument("--io-form-history", type=int, required=True, choices=sorted(EXPECTED_BY_IO_FORM))
    parser.add_argument("--netcdf-classic", required=True, choices=["yes", "no"],
                        help="use_netcdf_classic of the run (must match the namelist)")
    parser.add_argument("--darshan-logpath", required=True)
    parser.add_argument("--nodarshan", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-json", required=True)
    for name in ("app", "attempt", "case", "role", "kind", "jobid", "nodelist", "start-iso", "script",
                 "binary", "wrf-source", "darshan-config", "iosage-commit", "scratch-root"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--ntasks", type=int, required=True)
    parser.add_argument("--wall-s", type=float, required=True)
    parser.add_argument("--rc", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scratch = Path(args.scratch)
    namelist_text = Path(scratch, "namelist.input").read_text()
    values = parse_namelist(namelist_text)
    if namelist_int(values, "io_form_history") != args.io_form_history:
        logger.error("namelist io_form_history %s differs from the case value %s", values["io_form_history"],
                     args.io_form_history)
        sys.exit(2)
    classic = namelist_str(values, "use_netcdf_classic").lower().strip(".") in ("true", "t")
    if classic != (args.netcdf_classic == "yes"):
        logger.error("namelist use_netcdf_classic %s differs from the case value %s", values["use_netcdf_classic"],
                     args.netcdf_classic)
        sys.exit(2)
    rsl = parse_rsl(Path(scratch, "rsl.out.0000").read_text(errors="replace"))
    darshan_logs = sorted(str(p) for p in Path(args.darshan_logpath).glob(f"*id{args.jobid}-*"))
    darshan = None
    if not args.nodarshan:
        if len(darshan_logs) != 1:
            logger.error("expected one Darshan log for job %s, found %s", args.jobid, darshan_logs)
            sys.exit(2)
        darshan = darshan_records(darshan_logs[0])
        if darshan["nprocs"] != args.ntasks:
            logger.error("Darshan nprocs %d differs from ntasks %d", darshan["nprocs"], args.ntasks)
            sys.exit(2)
    correctness = build_correctness(rsl, values, scratch)
    names, frames, end, simulated_s = history_file_names(values)
    io_validation = build_io_validation(names, args.io_form_history, scratch, darshan, args.ntasks,
                                        {n: h["data_model"] for n, h in correctness["result"]["history"].items()},
                                        netcdf_classic=classic)
    normalized = "\n".join(line for line in namelist_text.splitlines() if not line.strip().startswith("io_form_history"))
    restart = Path(scratch, "wrfrst_d01_2019-11-26_23_00_00").resolve()
    boundary = Path(scratch, "wrfbdy_d01").resolve()
    source = Path(args.wrf_source)
    revision = subprocess.run(["git", "-C", str(source), "describe", "--tags", "--always", "--dirty"],
                              capture_output=True, text=True, check=False).stdout.strip()
    stripe = subprocess.run(["lfs", "getstripe", "-d", str(scratch)], capture_output=True, text=True, check=False)
    row = {
        "app": args.app, "attempt": args.attempt, "case": args.case, "role": args.role, "kind": args.kind,
        "repeat": args.repeat, "jobid": args.jobid, "nodes": args.nodes, "ntasks": args.ntasks,
        "nodelist": args.nodelist, "start_iso": args.start_iso, "wall_s": args.wall_s, "rc": args.rc,
        "nodarshan": args.nodarshan,
        "app_metric": {"name": "history_write_s", "value": rsl["history_write_s"]},
        "knobs": {"io_form_history": args.io_form_history, "history_interval_m": namelist_int(values, "history_interval_m"),
                  "use_netcdf_classic": classic},
        "work": {"forecast_start": date_string(values, "start"), "forecast_end": end,
                 "simulated_seconds": simulated_s, "time_step_s": namelist_int(values, "time_step"),
                 "expected_time_steps": simulated_s // namelist_int(values, "time_step"),
                 "history_interval_m": namelist_int(values, "history_interval_m"),
                 "expected_history_frames": frames, "frames_per_outfile": namelist_int(values, "frames_per_outfile"),
                 "grid_we": namelist_int(values, "e_we"), "grid_sn": namelist_int(values, "e_sn"),
                 "grid_vert": namelist_int(values, "e_vert"), "ranks": args.ntasks, "nodes": args.nodes,
                 "physics_suite": namelist_str(values, "physics_suite"),
                 "io_form_restart": namelist_int(values, "io_form_restart"),
                 "io_form_boundary": namelist_int(values, "io_form_boundary"), "use_netcdf_classic": classic,
                 "restart_sha256": sha256_file(restart), "boundary_sha256": sha256_file(boundary),
                 "namelist_normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()},
        "correctness": {k: correctness[k] for k in ("check", "pass", "result")},
        "io_validation": {k: io_validation[k] for k in ("check", "pass", "expected", "observed", "modules")},
        "out_files": sum(1 for n in names if Path(scratch, n).is_file()),
        "out_bytes": sum(Path(scratch, n).stat().st_size for n in names if Path(scratch, n).is_file()),
        "stripe": " ".join(stripe.stdout.split()) if stripe.returncode == 0 else None,
        "scratch_root": args.scratch_root, "darshan_logs": [] if args.nodarshan else darshan_logs,
        "darshan_conf_sha256": sha256_file(args.darshan_config) if Path(args.darshan_config).is_file() else None,
        "binary_sha256": sha256_file(args.binary), "script": args.script, "script_sha256": sha256_file(args.script),
        "iosage_commit": args.iosage_commit,
        "wrf_source_revision": revision,
        "wrf_local_patch_sha256": sha256_file(source / "frame" / "module_domain_type.F"),
        "namelist_sha256": hashlib.sha256(namelist_text.encode()).hexdigest(),
        "namelist_values": {k: values[k] for k in WORK_KEYS if k in values},
        "rsl": {k: rsl[k] for k in ("history_writes", "tables_computed", "completed_steps")},
    }
    evidence = {"row": row, "correctness_problems": correctness["problems"], "io_problems": io_validation["problems"]}
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
