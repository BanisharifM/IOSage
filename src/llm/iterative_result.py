"""Validation and loading for iterative optimization run records."""

from __future__ import annotations

import json
import math
from pathlib import Path


ITERATIVE_RESULT_VERSION = 2
TERMINAL_STATUSES = {
    "already_healthy",
    "converged",
    "plateau",
    "max_iterations_reached",
}


CONDITIONS = {
    "full",
    "no_ml",
    "no_kb",
    "no_shap",
    "single_shot",
    "no_feedback",
}


def result_id(workload, model, condition, run_id):
    """Return the stable identity used by result readers."""
    return f"{workload}:{model}:{condition}:{run_id}"


def validate_iterative_result(record):
    """Validate one completed primary run and return it unchanged."""
    if not isinstance(record, dict):
        raise ValueError("iterative result must be an object")
    if record.get("schema_version") != ITERATIVE_RESULT_VERSION:
        raise ValueError(f"iterative result schema must be {ITERATIVE_RESULT_VERSION}")
    for key in ("workload", "model", "condition", "result_id", "final_status"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"iterative result missing {key}")
    if record["condition"] not in CONDITIONS:
        raise ValueError(f"unknown iterative condition {record['condition']!r}")
    if (not isinstance(record.get("run_id"), (str, int))
            or isinstance(record.get("run_id"), bool)):
        raise ValueError("iterative result missing run_id")
    expected_id = result_id(
        record["workload"], record["model"], record["condition"], record["run_id"])
    if record["result_id"] != expected_id:
        raise ValueError("iterative result_id does not match its identity fields")
    if record["final_status"] not in TERMINAL_STATUSES:
        raise ValueError(f"iterative result has nonpassing status {record['final_status']!r}")
    if not isinstance(record.get("iterations"), list):
        raise ValueError("iterative result iterations must be a list")
    speedup = record.get("best_speedup")
    if (not isinstance(speedup, (int, float)) or isinstance(speedup, bool)
            or not math.isfinite(speedup) or speedup <= 0):
        raise ValueError("iterative result needs a positive finite best_speedup")
    if (not isinstance(record.get("total_iterations"), int)
            or isinstance(record.get("total_iterations"), bool)):
        raise ValueError("iterative total_iterations must be an integer")
    if (not isinstance(record.get("total_tokens"), int)
            or isinstance(record.get("total_tokens"), bool)):
        raise ValueError("iterative total_tokens must be an integer")
    for key in ("total_iterations", "total_cost_usd", "total_tokens",
                "total_execution_time_s"):
        value = record.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"iterative result needs nonnegative finite {key}")
    if record["total_iterations"] != len(record["iterations"]):
        raise ValueError("iterative total_iterations differs from its iteration list")
    config = record.get("config")
    expected_config = {"use_ml", "use_shap", "use_kb", "use_feedback", "dry_run"}
    if not isinstance(config, dict) or set(config) != expected_config:
        raise ValueError("iterative result config fields do not match the contract")
    if not all(isinstance(value, bool) for value in config.values()):
        raise ValueError("iterative result config flags must be booleans")
    if config["dry_run"]:
        raise ValueError("simulated iterative results are not primary evidence")
    if (not isinstance(record.get("max_iterations"), int)
            or isinstance(record.get("max_iterations"), bool)
            or record["max_iterations"] < 1):
        raise ValueError("iterative max_iterations must be a positive integer")
    for iteration in record["iterations"]:
        if not isinstance(iteration, dict):
            raise ValueError("iterative iteration must be an object")
        if iteration.get("executed") is True:
            speedup = iteration.get("speedup")
            if (not isinstance(speedup, (int, float)) or isinstance(speedup, bool)
                    or not math.isfinite(speedup) or speedup <= 0):
                raise ValueError("executed iteration needs a positive finite speedup")
    return record


def primary_result_files(results_dir):
    """Return candidate primary-run files while excluding configs and summaries."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"iterative results directory not found: {root}")
    candidates = []
    for path in sorted(root.glob("*.json")):
        name = path.name
        if name.endswith("_config.json") or "summary" in name or "complete_results" in name:
            continue
        if name.startswith(("iter_", "sweep_", "ablation_", "trackc_")):
            candidates.append(path)
    return candidates


def load_iterative_results(results_dir):
    """Load versioned primary runs and reject malformed files or duplicate IDs."""
    files = primary_result_files(results_dir)
    if not files:
        raise ValueError(f"no primary iterative result files in {results_dir}")
    records = []
    errors = []
    seen = set()
    for path in files:
        try:
            data = json.loads(path.read_text())
            items = data if isinstance(data, list) else [data]
            if not items:
                raise ValueError("empty result array")
            for item in items:
                validate_iterative_result(item)
                if item["result_id"] in seen:
                    raise ValueError(f"duplicate result_id {item['result_id']}")
                seen.add(item["result_id"])
                record = dict(item)
                record["_source_file"] = path.name
                records.append(record)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise ValueError("rejected iterative result files:\n" + "\n".join(errors))
    return records
