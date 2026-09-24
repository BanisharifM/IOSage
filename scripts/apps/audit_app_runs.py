"""Audit the application run manifests against the verification gates (EXECUTION_ROADMAP section 9).

Reads one manifest.jsonl (written by apps/common/run_lib.sh), groups the rows by (case, role,
control flag) and checks, per row and per group:
  1. exit code 0 and an explicit named correctness check that passed;
  2. a Darshan log for every traced run, which opens with PyDarshan;
  3. an explicit per-file I/O validation comparing like quantities;
  4. record cap not hit (PyDarshan partial flag);
  5. identical typed work signatures and preregistered correctness equivalence;
  6. at least the required number of repeats, and round-robin order (no case runs twice in a
     row within a series); verdicts through evaluate_candidate.
Every violation is listed; the exit code is 1 if any occurred. Nothing is filtered out.

Usage:
    python scripts/apps/audit_app_runs.py --manifest results/apps/nek5000/manifest.jsonl \
        --repeats 11 --pair N2:N2fix --pair N4:N5 --output results/apps/nek5000/audit.json
"""
import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.llm.closed_loop_metrics import aggregate_repeats, evaluate_candidate  # noqa: E402

logger = logging.getLogger("audit_app_runs")

WRITE_COUNTERS = {"POSIX": "POSIX_BYTES_WRITTEN", "STDIO": "STDIO_BYTES_WRITTEN"}


def correctness_passed(row):
    correctness = row.get("correctness")
    return (isinstance(correctness, dict)
            and isinstance(correctness.get("check"), str)
            and bool(correctness["check"].strip())
            and correctness.get("pass") is True)


def io_validation_passed(row):
    validation = row.get("io_validation")
    return (isinstance(validation, dict)
            and isinstance(validation.get("check"), str)
            and bool(validation["check"].strip())
            and validation.get("pass") is True)


def nested_value(value, dotted_name):
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_name)
        value = value[part]
    return value


def equivalent_results(problem, fixed, rule):
    if rule.get("mode") == "exact":
        return problem == fixed, None if problem == fixed else "correctness results differ"
    fields = rule.get("fields")
    if not isinstance(fields, dict) or not fields:
        return False, "equivalence rule must define mode=exact or numeric fields"
    for name, tolerance in fields.items():
        if not isinstance(tolerance, dict):
            return False, f"invalid tolerance for {name}"
        try:
            left = float(nested_value(problem, name))
            right = float(nested_value(fixed, name))
            absolute = float(tolerance.get("absolute", 0.0))
            relative = float(tolerance.get("relative", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"cannot compare {name}: {exc}"
        if not all(math.isfinite(v) for v in (left, right, absolute, relative)):
            return False, f"nonfinite value or tolerance for {name}"
        if absolute < 0 or relative < 0:
            return False, f"negative tolerance for {name}"
        if abs(right - left) > max(absolute, relative * abs(left)):
            return False, f"{name} differs: problem {left}, fix {right}"
    return True, None


def darshan_summary(path):
    """Bytes written (POSIX + STDIO), modules and partial flags of one Darshan log."""
    import darshan

    report = darshan.DarshanReport(str(path), read_all=True)
    written = 0
    for mod, col in WRITE_COUNTERS.items():
        if mod in report.records:
            written += int(report.records[mod].to_df()["counters"][col].clip(lower=0).sum())
    partial = sorted(m for m, v in report.modules.items() if v.get("partial_flag"))
    return {"written": written, "modules": sorted(report.modules), "partial": partial,
            "nprocs": int(report.metadata["job"]["nprocs"])}


def check_row(row, problems):
    key = f"{row.get('case')}/{row.get('role')} job {row.get('jobid')}"
    if row.get("rc") != 0:
        problems.append(f"{key}: exit code {row.get('rc')}")
    if not correctness_passed(row):
        problems.append(f"{key}: correctness check is missing, malformed, or failed")
    if row.get("nodarshan"):
        if row.get("darshan_logs"):
            problems.append(f"{key}: control run has a Darshan log")
        return
    logs = row.get("darshan_logs") or []
    if not logs:
        problems.append(f"{key}: no Darshan log")
        return
    written = 0
    for log in logs:
        try:
            info = darshan_summary(log)
        except Exception as exc:  # reported, never hidden
            problems.append(f"{key}: cannot open {Path(log).name}: {exc!r}")
            continue
        if info["partial"]:
            problems.append(f"{key}: record cap hit in {info['partial']}")
        written += info["written"]
    row["_darshan_written"] = written
    if not io_validation_passed(row):
        problems.append(f"{key}: per-file I/O validation is missing, malformed, or failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repeats", type=int, required=True, help="Required traced runs per (case, role)")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--pair", action="append", default=[], metavar="PROBLEM:FIX",
                        help="problem and fix case ids, optionally followed by the required verdict")
    parser.add_argument("--equivalence-config",
                        help="JSON file containing preregistered correctness rules under pairs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    equivalence = {}
    if args.pair:
        if not args.equivalence_config:
            parser.error("--equivalence-config is required when --pair is used")
        equivalence = json.loads(Path(args.equivalence_config).read_text()).get("pairs", {})

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if not rows:
        logger.error("manifest is empty: %s", args.manifest)
        sys.exit(2)
    problems = []
    for row in rows:
        check_row(row, problems)

    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("case"), row.get("role"), bool(row.get("nodarshan")))].append(row)
    summary = {}
    for (case, role, control), grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in grp if r.get("rc") == 0 and correctness_passed(r)
              and r.get("wall_s") and (control or io_validation_passed(r))]
        label = f"{case}/{role}" + ("/control" if control else "")
        if not control and len(ok) != args.repeats:
            problems.append(f"{label}: {len(ok)} usable runs, exactly {args.repeats} required")
        agg = aggregate_repeats([{"walltime_s": r["wall_s"], "write_bw_mb_s": 0.0,
                                  "bytes_total": r.get("out_bytes") or 0} for r in ok], args.confidence)
        summary[label] = {"runs": len(grp), "usable": len(ok), "jobids": [r.get("jobid") for r in grp],
                          "median_s": agg and round(agg["walltime_s"], 3),
                          "ci_s": agg and [round(agg["ci_lower_s"], 3), round(agg["ci_upper_s"], 3)],
                          "ci_coverage": agg and agg["ci_coverage"], "rel_mad": agg and round(agg["rel_mad"], 4),
                          "runs_s": agg and [round(w, 3) for w in agg["walltime_runs_s"]]}

    # round-robin: within the traced runs, sorted by start time, the same case never runs twice in a row
    traced = sorted((r for r in rows if not r.get("nodarshan") and r.get("start_iso")), key=lambda r: r["start_iso"])
    for prev, cur in zip(traced, traced[1:]):
        if (prev.get("case"), prev.get("role")) == (cur.get("case"), cur.get("role")) and len(groups) > 1:
            problems.append(f"order: {cur.get('case')}/{cur.get('role')} ran twice in a row "
                            f"(jobs {prev.get('jobid')}, {cur.get('jobid')})")
            break

    verdicts = {}
    for pair in args.pair:
        parts = pair.split(":")
        if len(parts) not in (2, 3):
            parser.error(f"invalid --pair value: {pair}")
        prob_id, fix_id = parts[:2]
        required_verdict = parts[2] if len(parts) == 3 else "faster"
        pair_key = f"{prob_id}:{fix_id}"
        rule = equivalence.get(pair_key)
        if not isinstance(rule, dict):
            problems.append(f"pair {pair_key}: no preregistered equivalence rule")
            continue
        prob = sorted((r for r in rows if r.get("case") == prob_id and not r.get("nodarshan")
                       and r.get("rc") == 0 and correctness_passed(r)), key=lambda r: r.get("repeat", -1))
        fix = sorted((r for r in rows if r.get("case") == fix_id and not r.get("nodarshan")
                      and r.get("rc") == 0 and correctness_passed(r)), key=lambda r: r.get("repeat", -1))
        if len(prob) != args.repeats or len(fix) != args.repeats:
            problems.append(f"pair {pair}: missing runs (problem {len(prob)}, fix {len(fix)})")
            continue
        work_note = None
        for baseline_row, fixed_row in zip(prob, fix):
            if (not isinstance(baseline_row.get("work"), dict) or not baseline_row["work"]
                    or not isinstance(fixed_row.get("work"), dict) or not fixed_row["work"]):
                work_note = "typed work signature is missing or malformed"
                break
            if baseline_row["work"] != fixed_row["work"]:
                work_note = "typed work signatures differ"
                break
            baseline_result = baseline_row["correctness"].get("result")
            fixed_result = fixed_row["correctness"].get("result")
            if baseline_result is None or fixed_result is None:
                work_note = "typed correctness result is missing"
                break
            same, reason = equivalent_results(baseline_result, fixed_result, rule)
            if not same:
                work_note = reason
                break
        if work_note:
            problems.append(f"pair {pair_key}: {work_note}")
        a = aggregate_repeats([{"walltime_s": r["wall_s"], "write_bw_mb_s": 0.0, "bytes_total": r.get("out_bytes") or 0}
                               for r in prob], args.confidence)
        b = aggregate_repeats([{"walltime_s": r["wall_s"], "write_bw_mb_s": 0.0, "bytes_total": r.get("out_bytes") or 0}
                               for r in fix], args.confidence)
        v = evaluate_candidate(a, b, best_speedup=1.0)
        verdicts[pair] = {"verdict": v["verdict"], "speedup": v["speedup"], "speedup_ci": v["speedup_ci"],
                          "problem_median_s": a["walltime_s"], "fix_median_s": b["walltime_s"], "work_note": work_note}
        if v["verdict"] != required_verdict:
            problems.append(f"pair {pair_key}: verdict {v['verdict']}, required {required_verdict}")
        logger.info("%s: %s, %.2fx, CI %s%s", pair, v["verdict"], v["speedup"], v["speedup_ci"],
                    f" [{work_note}]" if work_note else "")

    for p in problems:
        logger.warning("VIOLATION %s", p)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"manifest": str(args.manifest), "rows": len(rows), "repeats_required": args.repeats,
                               "groups": summary, "verdicts": verdicts, "violations": problems}, indent=2, default=str))
    logger.info("%d rows, %d groups, %d violations; wrote %s", len(rows), len(summary), len(problems), out)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
