#!/usr/bin/env python3
"""
Verify every benchmark sample against its constructed label.

For each sample of the label manifest (one log for IOR, mdtest, h5bench and
HACC-IO; the merged per-process logs of one job for DLIO and custom):
1. parse the log(s) and compute the engineered features
2. evaluate the label's rules from ``src.data.benchmark_verify`` (one
   observable rule per dimension; healthy must fail every bottleneck rule)
3. write one row per sample to the CSV report

Samples the manifest excludes (``source=none``) are listed with status
``excluded``. Exit status is 1 when any listed sample fails or cannot be
parsed, so the run cannot be mistaken for a pass.

Usage:
    python scripts/verify_all_ground_truth.py --report results/resubmission/verification/gt.csv
    python scripts/verify_all_ground_truth.py --bench-type custom
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.data.benchmark_logs import (  # noqa: E402
    AGGREGATED_BENCHMARKS, DEFAULT_MANIFEST, PER_RANK_BENCHMARKS, iter_benchmark_samples,
    load_manifest, manifest_row, posix_file_facts)
from src.data.benchmark_verify import DIMENSION_NAMES, verify_benchmark_log  # noqa: E402
from src.data.preprocessing import engineer_one  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = sorted(AGGREGATED_BENCHMARKS | PER_RANK_BENCHMARKS)
REPORT_COLUMNS = ["benchmark", "job_id", "n_files", "first_file", "scenario", "source",
                  "labels", "status", "cleaning_rule", "checks"]


def verify_benchmark(bench_type, log_dir, manifest):
    """One report row per sample of a benchmark."""
    rows = []
    for job_id, files, parsed in iter_benchmark_samples(bench_type, str(log_dir)):
        row = manifest_row(manifest, bench_type, job_id, files)
        base = {"benchmark": bench_type, "job_id": job_id, "n_files": len(files),
                "first_file": Path(files[0]).name}
        if row is None:
            rows.append(dict(base, scenario="", source="none", labels="", status="excluded",
                             cleaning_rule="", checks=""))
            continue
        labels = {d: int(row[d]) for d in DIMENSION_NAMES}
        base.update(scenario=row["scenario"], source=row["source"],
                    labels=",".join(d for d in DIMENSION_NAMES if labels[d]))
        if parsed is None:
            rows.append(dict(base, status="unparsed", cleaning_rule="", checks=""))
            continue
        features = engineer_one(parsed)
        context = dict(log_paths=files, **posix_file_facts(files))
        passed, report = verify_benchmark_log(features, labels, context)
        checks = "; ".join(f"{name}={c['status']} ({c['value']})" for name, c in report["checks"].items())
        cleaning = "pass" if report["cleaning_rule"] else "below: " + report["cleaning_reason"]
        rows.append(dict(base, status="pass" if passed else "fail", cleaning_rule=cleaning, checks=checks))
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pass", "fail", "unparsed", "excluded")}
    counts["below_cleaning_rule"] = sum(r["cleaning_rule"].startswith("below") for r in rows)
    logger.info("  %s: %s", bench_type, counts)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark samples against their labels")
    parser.add_argument("--log-dir", default=str(PROJECT_DIR / "data" / "benchmark_logs"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--bench-type", choices=["all"] + BENCHMARKS, default="all")
    parser.add_argument("--report", help="CSV with one row per sample")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    bench_types = BENCHMARKS if args.bench_type == "all" else [args.bench_type]
    rows = []
    for bench in bench_types:
        log_dir = Path(args.log_dir) / bench
        if not log_dir.is_dir():
            raise FileNotFoundError(f"log directory not found: {log_dir}")
        logger.info("Verifying %s", bench)
        rows.extend(verify_benchmark(bench, log_dir, manifest))

    selected_manifest = manifest[manifest["benchmark"].isin(bench_types)].copy()
    if selected_manifest.empty:
        raise ValueError(f"manifest has no samples for {bench_types}")
    expected_keys = {
        (
            row.benchmark,
            str(row.job_id),
            row.log_file if row.benchmark in AGGREGATED_BENCHMARKS else "",
        )
        for row in selected_manifest.itertuples(index=False)
    }
    actual_keys = {
        (
            row["benchmark"],
            str(row["job_id"]),
            row["first_file"] if row["benchmark"] in AGGREGATED_BENCHMARKS else "",
        )
        for row in rows
    }
    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        detail = next(iter(sorted(missing or extra)))
        raise RuntimeError(
            "verified sample set differs from the label manifest: "
            f"missing={len(missing)}, extra={len(extra)}, first={detail}"
        )

    totals = {s: sum(r["status"] == s for r in rows) for s in ("pass", "fail", "unparsed", "excluded")}
    totals["below_cleaning_rule"] = sum(r["cleaning_rule"].startswith("below") for r in rows)
    logger.info("OVERALL: %s", totals)
    for r in rows:
        if r["status"] == "fail":
            logger.info("  FAIL %s %s [%s]: %s", r["benchmark"], r["job_id"], r["labels"], r["checks"])

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Report written: %s (%d samples)", args.report, len(rows))

    return 1 if totals["fail"] or totals["unparsed"] else 0


if __name__ == "__main__":
    sys.exit(main())
