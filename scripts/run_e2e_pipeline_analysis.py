#!/usr/bin/env python3
"""Analyze one declared E2E before/after pair from its job manifest."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.parse_darshan import parse_darshan_log
from src.data.preprocessing import engineer_one
from src.ioprescriber.contracts import validate_pipeline_result
from src.ioprescriber.pipeline import IOPrescriber

MANIFEST_FIELDS = {
    "config_id", "fix_id", "job_id", "executable", "output",
    "darshan_log", "completed_utc",
}
CORRECTNESS_FIELDS = {
    "before_config", "after_config", "job_id", "before_output", "after_output",
    "checker", "passed", "completed_utc",
}


def load_manifest(path):
    manifest = Path(path).resolve()
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("E2E manifest fields do not match the contract")
        rows = list(reader)
    if not rows:
        raise ValueError("E2E manifest has no runs")
    identities = [(row["config_id"], row["job_id"], row["darshan_log"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("E2E manifest has duplicate run identities")
    for row in rows:
        for key in MANIFEST_FIELDS:
            if not row[key]:
                raise ValueError(f"E2E manifest row has empty {key}")
        log_path = Path(row["darshan_log"]).resolve()
        executable = Path(row["executable"]).name
        if not log_path.is_file() or not log_path.name.endswith(".darshan"):
            raise FileNotFoundError(f"declared Darshan log is missing: {log_path}")
        if executable not in log_path.name or f"id{row['job_id']}" not in log_path.name:
            raise ValueError("Darshan path does not match the declared executable and job")
        row["darshan_log"] = str(log_path)
    return manifest, rows


def select_one(rows, config_id, job_id=None):
    matches = [row for row in rows if row["config_id"] == config_id
               and (job_id is None or row["job_id"] == job_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one declared {config_id} run, found {len(matches)}")
    return matches[0]


def load_correctness(path):
    correctness_path = Path(path).resolve()
    with correctness_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or ()) != CORRECTNESS_FIELDS:
            raise ValueError("E2E correctness fields do not match the contract")
        rows = list(reader)
    if not rows:
        raise ValueError("E2E correctness manifest has no comparisons")
    for row in rows:
        if any(not row[key] for key in CORRECTNESS_FIELDS):
            raise ValueError("E2E correctness row has an empty field")
        if row["checker"] != "h5diff --quiet" or row["passed"] != "true":
            raise ValueError("E2E correctness comparison did not pass")
    return correctness_path, rows


def select_correctness(rows, before, after):
    matches = [row for row in rows
               if row["before_config"] == before["config_id"]
               and row["after_config"] == after["config_id"]
               and row["job_id"] == before["job_id"]
               and row["before_output"] == before["output"]
               and row["after_output"] == after["output"]]
    if len(matches) != 1:
        raise ValueError(
            f"expected one matching correctness comparison, found {len(matches)}")
    return matches[0]


def features_for(row):
    parsed = parse_darshan_log(row["darshan_log"], strict=True)
    features = engineer_one(parsed)
    runtime = float(features["runtime_seconds"])
    total_bytes = float(features["POSIX_BYTES_READ"] + features["POSIX_BYTES_WRITTEN"])
    if not math.isfinite(runtime) or runtime <= 0 or total_bytes <= 0:
        raise ValueError(f"invalid measured work for {row['config_id']}")
    return features, runtime, total_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--correctness-manifest", required=True)
    parser.add_argument("--before-config", default="pathological")
    parser.add_argument("--after-config", default="ultra_optimized")
    parser.add_argument("--job-id")
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--expected-fix-id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest, rows = load_manifest(args.manifest)
    before = select_one(rows, args.before_config, args.job_id)
    after = select_one(rows, args.after_config, args.job_id)
    if before["job_id"] != after["job_id"]:
        raise ValueError("before and after runs must come from one declared job")
    correctness_path, correctness_rows = load_correctness(args.correctness_manifest)
    correctness = select_correctness(correctness_rows, before, after)
    expected_fix = args.expected_fix_id or after["fix_id"]
    if expected_fix in ("", "none"):
        raise ValueError("after run must declare a structured fix_id")

    before_features, before_runtime, before_bytes = features_for(before)
    after_features, after_runtime, after_bytes = features_for(after)
    pipeline = IOPrescriber(args.model_bundle, args.knowledge_base,
                            llm_model="claude-sonnet", use_shap=True)
    analysis = pipeline.analyze(
        before_features,
        workload_name=f"e2e_{before['config_id']}",
        sample_id=f"e2e/{before['job_id']}/{Path(before['darshan_log']).name}",
        job_group=f"e2e/{before['job_id']}",
    )
    validate_pipeline_result(analysis)
    parsed = analysis["recommendation"]["parsed"]
    if not isinstance(parsed, dict):
        raise ValueError("E2E recommendation is absent")
    fix_ids = {item.get("evidence_fix_id") for item in parsed.get("recommendations", [])}
    if expected_fix not in fix_ids:
        raise ValueError(f"recommendation does not cite expected fix_id {expected_fix}")

    result = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "correctness_manifest": str(correctness_path),
        "correctness_manifest_sha256": hashlib.sha256(
            correctness_path.read_bytes()).hexdigest(),
        "correctness": correctness,
        "job_id": before["job_id"],
        "before": before,
        "after": after,
        "before_io_bytes": before_bytes,
        "after_io_bytes": after_bytes,
        "before_runtime_s": before_runtime,
        "after_runtime_s": after_runtime,
        "walltime_speedup": before_runtime / after_runtime,
        "expected_fix_id": expected_fix,
        "pipeline_result": analysis,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
