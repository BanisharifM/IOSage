#!/usr/bin/env python3
"""
Comprehensive TraceBench evaluation: compare IOPrescriber vs IONavigator vs Drishti.

Parses ION diagnosis text and Drishti output for each TraceBench real-app trace,
maps their detected issues to our 8-dimension taxonomy, and computes per-system
precision/recall/F1 against TraceBench ground truth.

Also includes Iterative context: what our iterative optimizer would do on these traces.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TRACEBENCH = PROJECT / "data" / "external" / "tracebench" / "TraceBench"
LLMEVAL = TRACEBENCH / "LLMEval"
LABELS_FILE = TRACEBENCH / "Datasets" / "real_app_bench" / "trace_labels.json"
OUR_RESULTS = PROJECT / "results" / "e2e_evaluation" / "tracebench_real_app_results.json"
TRACKC_SUMMARY = PROJECT / "results" / "iterative" / "trackc_full_summary.json"
OUTPUT = PROJECT / "results" / "e2e_evaluation" / "tracebench_comprehensive.json"

# TraceBench label -> our dimension mapping
LABEL_MAP = {
    "SML-R": "access_granularity",
    "SML-W": "access_granularity",
    "MSL-R": "access_granularity",
    "MSL-W": "access_granularity",
    "HMD": "metadata_intensity",
    "SLIM": "parallelism_efficiency",
    "RLIM": "parallelism_efficiency",
    "RMA-R": "access_pattern",
    "RMA-W": "access_pattern",
    "RDA-R": "access_pattern",
    "NC-R": "interface_choice",
    "NC-W": "interface_choice",
    "LLL-R": "interface_choice",
    "LLL-W": "interface_choice",
    "MPNM": "interface_choice",
    "SHF": "file_strategy",
}

# The 9 real-app trace keys (ordered)
TRACE_KEYS = [
    "(AMReX)_jeanbez_h5bench_amrex_sync_id15672431-1118360_9-15-38432-5714886783581687674_1",
    "(E2E_Baseline)_original_jeanbez_write_3d_nc4_id45139176_8-5-1057-1043768654573709768_1628147943",
    "(E2E_Optimized)_optimized_jeanbez_write_3d_nc4_id45138874_8-5-631-13015627732142842593_1628147442",
    "(H5_Bench)_brtnfld_h5bench_write_id1363501_8-26-63389-8217696575051336807_84",
    "(OpenPMD_Baseline)_baseline_jlbez_8a_benchmark_write_parallel_id1321662_8-21-5892-15802854900629188750_106",
    "(OpenPMD_Optimized)_optimized_jlbez_8a_benchmark_write_parallel_id1322696_8-21-14519-8141979180909667175_12",
    "(Optimize-MP)_houhun_sw4_id56546140-264967_3-21-63846-13415404289583628136_1",
    "(ROBL_IOR)_robl_ior_id614345_8-8-66546-14494205032741333235_1",
    "(Treb_ViscousDriver3d)_treb_viscousDriver3d",
]

TRACE_NAMES = [
    "AMReX",
    "E2E Baseline",
    "E2E Optimized",
    "H5 Bench",
    "OpenPMD Baseline",
    "OpenPMD Optimized",
    "SW4/Optimize-MP",
    "ROBL IOR",
    "Treb ViscousDriver3d",
]

ALL_DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
]


def parse_ion_detections(text):
    """Parse ION-1.0 diagnosis text and extract detected dimensions.

    ION 1.0 discusses ALL issue categories for every trace, often saying
    "no issue" or "negligible" for categories that are not problematic.
    We must parse per-paragraph and check for negation patterns.
    """
    if not text:
        return set()

    detected = set()

    # Split into paragraphs (ION uses numbered/bulleted items)
    paragraphs = re.split(r'\n\s*\n|\n\s{4,}', text)
    # Also try splitting on numbered items
    items = re.split(r'\n\s+(?=\d+\.|\*|•|–)', text)
    # Combine and also split on category headers
    all_sections = []
    for chunk in paragraphs + items:
        # Further split on "Category:" patterns
        sub = re.split(r'(?=(?:Small I/O|Random I/O|Misaligned I/O|Non-Collective|'
                       r'Load Imbalanc|High Metadata|Shared File|Low.?Level))',
                       chunk, flags=re.IGNORECASE)
        all_sections.extend(sub)

    # Negative patterns that indicate "not an issue"
    NEGATION = re.compile(
        r'no\s+issue|not\s+(?:a\s+)?(?:significant|major|concern)|negligible|'
        r'no\s+concern|unlikely\s+to\s+be|not\s+a\s+widespread|'
        r'no\s+issues|effective|highlighting\s+no|indicating\s+no|'
        r'not\s+significant|insignificant|no\s+small\s+i/o\s+issue|'
        r'not\s+a\s+concern|no\s+load\s+imbalance',
        re.IGNORECASE
    )

    def section_is_positive(section_text):
        """Check if a section flags an issue as positive (problematic)."""
        lower = section_text.lower()
        # If negation found, it's not a real detection
        if NEGATION.search(lower):
            return False
        # Must have some positive indicator
        positive = re.search(
            r'significant|issue|concern|inefficien|impact|problem|'
            r'severe|notable|potential|high|substantial|important|'
            r'should be|needs?\s+address|clear\s+load\s+imbalance|'
            r'performance\s+degradation',
            lower
        )
        return bool(positive)

    for section in all_sections:
        sec_lower = section.lower()

        # Small I/O -> access_granularity
        if re.search(r'small\s+i/o', sec_lower):
            if section_is_positive(section):
                detected.add("access_granularity")

        # Misaligned I/O -> access_granularity
        if re.search(r'misaligned\s+i/o|misaligned\s+file', sec_lower):
            if section_is_positive(section):
                detected.add("access_granularity")

        # Random I/O -> access_pattern
        if re.search(r'random\s+i/o|random\s+access', sec_lower):
            if section_is_positive(section):
                detected.add("access_pattern")

        # Non-Collective I/O -> interface_choice
        if re.search(r'non-collective|collective\s+i/o|collective\s+read|collective\s+write', sec_lower):
            if section_is_positive(section):
                detected.add("interface_choice")

        # Low Level Library -> interface_choice
        if re.search(r'low.?level\s+library', sec_lower):
            if section_is_positive(section):
                detected.add("interface_choice")

        # Load Imbalanced I/O -> parallelism_efficiency
        if re.search(r'load\s+imbalanc', sec_lower):
            if section_is_positive(section):
                detected.add("parallelism_efficiency")

        # High Metadata I/O -> metadata_intensity
        if re.search(r'(?:high\s+)?metadata\s+i/o|metadata\s+operations?', sec_lower):
            if section_is_positive(section):
                detected.add("metadata_intensity")

        # Shared File I/O -> file_strategy
        if re.search(r'shared\s+file', sec_lower):
            if section_is_positive(section):
                detected.add("file_strategy")

    # Also check the Summary section specifically (usually at the end)
    summary_match = re.search(r'Summary.*$', text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group().lower()
        if re.search(r'small\s+i/o|small.*operations', summary):
            detected.add("access_granularity")
        if re.search(r'random\s+i/o|random.*access', summary):
            detected.add("access_pattern")
        if re.search(r'load\s+imbalanc', summary):
            detected.add("parallelism_efficiency")
        if re.search(r'shared\s+file', summary):
            detected.add("file_strategy")
        if re.search(r'metadata', summary):
            detected.add("metadata_intensity")
        if re.search(r'non-collective|collective', summary):
            detected.add("interface_choice")
        if re.search(r'misaligned', summary):
            detected.add("access_granularity")

    return detected


def parse_drishti_detections(text):
    """Parse Drishti output and extract detected dimensions."""
    if not text:
        return set()

    detected = set()

    # Small read/write requests -> access_granularity
    if re.search(r"small (read|write) requests", text):
        detected.add("access_granularity")

    # Misaligned -> access_granularity
    if re.search(r"misaligned file requests", text):
        detected.add("access_granularity")

    # Load imbalance -> parallelism_efficiency
    if re.search(r"load imbalance.*detected|Load imbalance", text, re.IGNORECASE):
        detected.add("parallelism_efficiency")

    # Non-collective -> interface_choice
    if re.search(r"does not use collective (read|write) operations", text):
        detected.add("interface_choice")

    # Shared file small requests -> file_strategy
    if re.search(r"small (read|write) requests to a shared file", text):
        detected.add("file_strategy")

    return detected


def compute_metrics(detected, ground_truth, all_dims):
    """Compute TP/FP/FN/TN against ground truth."""
    tp, fp, fn, tn = 0, 0, 0, 0
    tp_dims, fp_dims, fn_dims = [], [], []

    for dim in all_dims:
        in_gt = dim in ground_truth
        in_det = dim in detected
        if in_gt and in_det:
            tp += 1
            tp_dims.append(dim)
        elif not in_gt and in_det:
            fp += 1
            fp_dims.append(dim)
        elif in_gt and not in_det:
            fn += 1
            fn_dims.append(dim)
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tp_dims": tp_dims, "fp_dims": fp_dims, "fn_dims": fn_dims,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    # Load ground truth
    with open(LABELS_FILE) as f:
        gt_labels = json.load(f)

    # Load our results
    with open(OUR_RESULTS) as f:
        our_results = json.load(f)

    # Load Iterative summary
    trackc = {}
    if TRACKC_SUMMARY.exists():
        with open(TRACKC_SUMMARY) as f:
            trackc = json.load(f)

    # Build per-trace comparison
    comparison = []
    agg = {
        "ioprescriber": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "ion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "drishti": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    per_dim = {
        system: {dim: {"tp": 0, "fp": 0, "fn": 0} for dim in ALL_DIMS}
        for system in ["ioprescriber", "ion", "drishti"]
    }

    for i, trace_key in enumerate(TRACE_KEYS):
        trace_name = TRACE_NAMES[i]

        # Ground truth
        raw_labels = gt_labels.get(trace_key, [])
        gt_dims = set()
        for lbl in raw_labels:
            if lbl in LABEL_MAP:
                gt_dims.add(LABEL_MAP[lbl])

        # Our ML detections -- match by trace_name or partial file_key match
        our_trace = None
        for t in our_results["per_trace_results"]:
            if t["file_key"] == trace_key:
                our_trace = t
                break
            if t.get("trace_name") == trace_name:
                our_trace = t
                break
            # Partial match: our key may be longer (Treb has full exe name)
            if trace_key in t["file_key"] or t["file_key"].startswith(trace_key):
                our_trace = t
                break

        our_detected = set()
        if our_trace:
            our_detected = set(our_trace.get("ml_detected", []))
            # Remove 'healthy' if present
            our_detected.discard("healthy")

        # ION detections
        ion_file = LLMEVAL / trace_key / "ION-1.0_diagnosis.txt"
        ion_text = ""
        if ion_file.exists():
            ion_text = ion_file.read_text()
        ion_detected = parse_ion_detections(ion_text)

        # Drishti detections
        drishti_file = LLMEVAL / trace_key / "drishti.txt"
        drishti_text = ""
        if drishti_file.exists():
            drishti_text = drishti_file.read_text()
        drishti_detected = parse_drishti_detections(drishti_text)

        # Compute metrics for each system
        our_metrics = compute_metrics(our_detected, gt_dims, ALL_DIMS)
        ion_metrics = compute_metrics(ion_detected, gt_dims, ALL_DIMS)
        drishti_metrics = compute_metrics(drishti_detected, gt_dims, ALL_DIMS)

        # Aggregate
        for system, metrics in [("ioprescriber", our_metrics), ("ion", ion_metrics), ("drishti", drishti_metrics)]:
            for k in ["tp", "fp", "fn", "tn"]:
                agg[system][k] += metrics[k]

        # Per-dimension tracking
        for system, detected_set, metrics in [
            ("ioprescriber", our_detected, our_metrics),
            ("ion", ion_detected, ion_metrics),
            ("drishti", drishti_detected, drishti_metrics),
        ]:
            for dim in metrics["tp_dims"]:
                per_dim[system][dim]["tp"] += 1
            for dim in metrics["fp_dims"]:
                per_dim[system][dim]["fp"] += 1
            for dim in metrics["fn_dims"]:
                per_dim[system][dim]["fn"] += 1

        # LLM recs and groundedness from our results
        n_recs = 0
        n_grounded = 0
        if our_trace and "llm_result" in our_trace:
            llm = our_trace["llm_result"]
            recs = llm.get("recommendations", [])
            n_recs = len(recs)
            n_grounded = sum(1 for r in recs if r.get("grounded", False))

        entry = {
            "trace_name": trace_name,
            "trace_key": trace_key,
            "tracebench_labels": raw_labels,
            "ground_truth_dims": sorted(gt_dims),
            "ioprescriber": {
                "detected": sorted(our_detected),
                **our_metrics,
                "n_recommendations": n_recs,
                "n_grounded": n_grounded,
            },
            "ion_1_0": {
                "detected": sorted(ion_detected),
                **ion_metrics,
                "has_diagnosis": bool(ion_text.strip()),
            },
            "drishti": {
                "detected": sorted(drishti_detected),
                **drishti_metrics,
                "has_output": bool(drishti_text.strip()),
            },
        }
        comparison.append(entry)

    # Compute aggregate metrics
    agg_metrics = {}
    for system in ["ioprescriber", "ion", "drishti"]:
        tp, fp, fn = agg[system]["tp"], agg[system]["fp"], agg[system]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        agg_metrics[system] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": agg[system]["tn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        }

    # Per-dimension F1 for each system
    per_dim_f1 = {}
    for system in ["ioprescriber", "ion", "drishti"]:
        per_dim_f1[system] = {}
        for dim in ALL_DIMS:
            tp = per_dim[system][dim]["tp"]
            fp = per_dim[system][dim]["fp"]
            fn = per_dim[system][dim]["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_dim_f1[system][dim] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
            }

    # Iterative context
    trackc_context = {
        "applicable": False,
        "reason": "Iterative requires SLURM job submission and new Darshan log collection. "
                  "TraceBench traces are from external HPC systems (Summit, Theta, Cori) "
                  "and cannot be re-executed on Delta. Iterative is validated separately "
                  "using our own benchmark workloads.",
        "trackc_summary": {},
    }
    if trackc:
        trackc_context["trackc_summary"] = {
            "total_runs": trackc.get("total_runs", 0),
            "per_model_geomean_speedup": {
                m: d.get("geomean", 0) for m, d in trackc.get("per_model", {}).items()
            },
            "best_model": max(
                trackc.get("per_model", {}).items(),
                key=lambda x: x[1].get("geomean", 0),
                default=("none", {})
            )[0],
            "best_geomean": max(
                (d.get("geomean", 0) for d in trackc.get("per_model", {}).values()),
                default=0,
            ),
        }
        # Map Iterative workloads to TraceBench dimensions
        trackc_context["dimension_coverage"] = {
            "ior_fsync_heavy": ["throughput_utilization"],
            "ior_small_posix": ["access_granularity"],
            "ior_small_direct": ["access_granularity"],
            "ior_random_access": ["access_pattern"],
            "ior_misaligned": ["access_granularity"],
            "ior_interface_shared": ["interface_choice", "file_strategy"],
            "mdtest_metadata_storm": ["metadata_intensity"],
            "ior_healthy_baseline": ["healthy"],
        }

    # Groundedness from our results
    total_recs = our_results["aggregate_metrics"]["groundedness_total"]
    grounded_recs = our_results["aggregate_metrics"]["groundedness_grounded"]

    output = {
        "metadata": {
            "date": "2026-03-26",
            "description": "Comprehensive TraceBench evaluation comparing IOPrescriber (Single-shot) "
                          "vs IONavigator 1.0 vs Drishti on 9 real-application Darshan traces",
            "n_traces": 9,
            "n_dimensions": 7,
            "note": "TraceBench does not label throughput_utilization or temporal_pattern. "
                    "FPs in those dimensions may reflect valid detections not in GT.",
        },
        "aggregate_comparison": agg_metrics,
        "per_dimension_comparison": per_dim_f1,
        "per_trace_comparison": comparison,
        "ioprescriber_llm_quality": {
            "total_recommendations": total_recs,
            "grounded_recommendations": grounded_recs,
            "groundedness_score": round(grounded_recs / total_recs, 4) if total_recs > 0 else 0,
            "nc_nofill_gap": True,
        },
        "track_c_context": trackc_context,
    }

    # Save
    os.makedirs(OUTPUT.parent, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("=" * 80)
    print("TraceBench Comprehensive Evaluation Summary")
    print("=" * 80)
    print(f"\n{'System':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 65)
    for system in ["ioprescriber", "ion", "drishti"]:
        m = agg_metrics[system]
        print(f"{system:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print(f"\n{'Per-Dimension F1':}")
    print(f"{'Dimension':<25} {'IOPrescriber':>12} {'ION':>12} {'Drishti':>12}")
    print("-" * 65)
    for dim in ALL_DIMS:
        io_f1 = per_dim_f1["ioprescriber"][dim]["f1"]
        ion_f1 = per_dim_f1["ion"][dim]["f1"]
        dr_f1 = per_dim_f1["drishti"][dim]["f1"]
        print(f"{dim:<25} {io_f1:>12.3f} {ion_f1:>12.3f} {dr_f1:>12.3f}")

    print(f"\nPer-Trace Detection Comparison:")
    print(f"{'Trace':<22} {'GT Dims':<35} {'IOPrescriber':<25} {'ION':<25} {'Drishti':<25}")
    print("-" * 135)
    for entry in comparison:
        gt = ", ".join(entry["ground_truth_dims"]) or "(none)"
        ours = ", ".join(entry["ioprescriber"]["detected"]) or "(healthy)"
        ion = ", ".join(entry["ion_1_0"]["detected"]) or "(none)"
        dri = ", ".join(entry["drishti"]["detected"]) or "(none)"
        print(f"{entry['trace_name']:<22} {gt:<35} {ours:<25} {ion:<25} {dri:<25}")

    print(f"\nIOPrescriber LLM Quality: {grounded_recs}/{total_recs} recommendations grounded ({output['ioprescriber_llm_quality']['groundedness_score']:.1%})")

    if trackc:
        best = trackc_context["trackc_summary"]["best_model"]
        geomean = trackc_context["trackc_summary"]["best_geomean"]
        print(f"\nIterative (separate validation): Best model={best}, geomean speedup={geomean:.2f}x across {trackc.get('total_runs', 0)} runs")
        print("Iterative not applicable to TraceBench (requires SLURM re-execution)")

    print(f"\nResults saved to: {OUTPUT}")
    return output


if __name__ == "__main__":
    main()
