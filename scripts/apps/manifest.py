"""Manifest row contract for the application runs.

One JSON line per run in ``manifest.jsonl``. The row carries four typed groups that the
audit (``scripts/apps/audit_app_runs.py``) reads:

* ``work``: the scientific work of the run (steps, checkpoints, precision, ranks, mesh and
  input checksums). Identical between a problem case and its fix.
* ``correctness``: a named check, its Boolean result and the typed physical ``result`` the
  preregistered equivalence rule compares between problem and fix.
* ``io_validation``: a named check, its Boolean result, the ``expected`` file organization,
  the ``observed`` organization, and the Darshan ``modules`` seen.
* provenance: case, role, kind, repeat, job id, nodes, tasks, node list, start time, wall
  seconds, exit code, control flag, commit, script and binary hashes, Darshan config hash and
  Darshan log paths.

``validate_row`` returns every contract violation of a row; ``append_row`` refuses to write a
row that has any.
"""
import json
from pathlib import Path

REQUIRED = {
    "app": str, "attempt": str, "case": str, "role": str, "kind": str, "repeat": int, "jobid": str,
    "nodes": int, "ntasks": int, "nodelist": str, "start_iso": str, "wall_s": float, "rc": int,
    "nodarshan": bool, "app_metric": dict, "knobs": dict, "work": dict, "correctness": dict,
    "io_validation": dict, "out_files": int, "out_bytes": int, "scratch_root": str,
    "darshan_logs": list, "binary_sha256": str, "script": str, "script_sha256": str,
    "iosage_commit": str,
}
OPTIONAL = {"stripe": str, "darshan_conf_sha256": str}
ROLES = {"problem", "fix", "abstain", "null_A", "null_A_prime", "smoke"}
KINDS = {"natural", "constructed", "null_check", "smoke"}


def _named_check(value, name, problems, extra_keys=()):
    if not isinstance(value, dict):
        problems.append(f"{name} is not an object")
        return
    check = value.get("check")
    if not isinstance(check, str) or not check.strip():
        problems.append(f"{name}.check is not a non-empty string")
    if not isinstance(value.get("pass"), bool):
        problems.append(f"{name}.pass is not a boolean")
    for key in extra_keys:
        if not isinstance(value.get(key), dict) or not value[key]:
            problems.append(f"{name}.{key} is not a non-empty object")


def validate_row(row):
    """Every contract violation of one manifest row (empty list when it conforms)."""
    problems = []
    if not isinstance(row, dict):
        return ["row is not an object"]
    for key, kind in REQUIRED.items():
        if key not in row:
            problems.append(f"missing {key}")
        elif kind is float:
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
                problems.append(f"{key} is not a number")
        elif kind is int:
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                problems.append(f"{key} is not an integer")
        elif not isinstance(row[key], kind):
            problems.append(f"{key} is not {kind.__name__}")
    for key, kind in OPTIONAL.items():
        if row.get(key) is not None and not isinstance(row[key], kind):
            problems.append(f"{key} is not {kind.__name__}")
    if problems:
        return problems
    if row["role"] not in ROLES:
        problems.append(f"role {row['role']!r} not in {sorted(ROLES)}")
    if row["kind"] not in KINDS:
        problems.append(f"kind {row['kind']!r} not in {sorted(KINDS)}")
    if row["repeat"] < 0:
        problems.append("repeat is negative")
    if not row["work"]:
        problems.append("work is empty")
    for key, value in row["work"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            problems.append(f"work.{key} is not a typed scalar")
    _named_check(row["correctness"], "correctness", problems, extra_keys=("result",))
    _named_check(row["io_validation"], "io_validation", problems, extra_keys=("expected", "observed"))
    if isinstance(row["io_validation"], dict) and not isinstance(row["io_validation"].get("modules"), list):
        problems.append("io_validation.modules is not a list")
    if not all(isinstance(p, str) and p for p in row["darshan_logs"]):
        problems.append("darshan_logs holds a non-string entry")
    if row["nodarshan"] and row["darshan_logs"]:
        problems.append("control run lists Darshan logs")
    if not row["nodarshan"] and not row["darshan_logs"]:
        problems.append("traced run lists no Darshan log")
    metric = row["app_metric"]
    if not isinstance(metric.get("name"), str) or isinstance(metric.get("value"), bool) \
            or not isinstance(metric.get("value"), (int, float)):
        problems.append("app_metric needs a name and a numeric value")
    return problems


def append_row(path, row):
    """Validate ``row`` and append it as one JSON line; raises ValueError on a violation."""
    problems = validate_row(row)
    if problems:
        raise ValueError("manifest row violates the contract: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
