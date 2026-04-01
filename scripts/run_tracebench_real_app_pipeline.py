"""
Run IOPrescriber pipeline on TraceBench real application Darshan logs.

Tier 1 of E2E validation: evaluate recommendation accuracy on real
applications with known ground-truth bottleneck labels from TraceBench.

Key traces:
  - E2E Baseline / E2E Optimized (Bez et al., ISC 2023 NC_NOFILL fix)
  - OpenPMD Baseline / Optimized (particle physics simulation)
  - AMReX (adaptive mesh refinement)
  - SW4 / Optimize-MP (seismic simulation)
  - ROBL_IOR, H5 Bench, Treb_ViscousDriver3d

Usage:
    source .env && python scripts/run_tracebench_real_app_pipeline.py

Outputs:
    results/e2e_evaluation/tracebench_real_app_results.json
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Load .env
env_path = PROJECT_DIR / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                key = key.replace("export ", "").strip()
                val = val.strip().strip('"')
                os.environ[key] = val

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tracebench_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DARSHAN_DIR = (
    PROJECT_DIR
    / "data"
    / "external"
    / "tracebench"
    / "TraceBench"
    / "Datasets"
    / "real_app_bench"
    / "darshan_files"
    / "darshan"
)

LABEL_FILE = (
    PROJECT_DIR
    / "data"
    / "external"
    / "tracebench"
    / "TraceBench"
    / "Datasets"
    / "real_app_bench"
    / "trace_labels.json"
)

LABEL_MAPPING_FILE = (
    PROJECT_DIR / "data" / "external" / "tracebench" / "label_mapping.json"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "e2e_evaluation"

OUR_DIMENSIONS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
]


def load_tracebench_labels():
    """Load TraceBench labels and mapping, return per-trace ground-truth
    mapped to our 8-dimension taxonomy."""
    with open(LABEL_FILE) as f:
        raw_labels = json.load(f)
    with open(LABEL_MAPPING_FILE) as f:
        mapping = json.load(f)

    tb_to_dim = {}
    for tb_label, info in mapping["tracebench_to_our_taxonomy"].items():
        dim = info["our_dimension"]
        if dim is not None:
            tb_to_dim[tb_label] = dim

    per_trace = {}
    for trace_key, tb_labels in raw_labels.items():
        dims_active = set()
        tb_details = []
        for lbl in tb_labels:
            dim = tb_to_dim.get(lbl)
            if dim:
                dims_active.add(dim)
                tb_details.append({"tracebench_label": lbl, "mapped_dim": dim})
        per_trace[trace_key] = {
            "tracebench_labels": tb_labels,
            "mapped_dims": sorted(dims_active),
            "mapping_details": tb_details,
        }
    return per_trace


def parse_darshan_subprocess(darshan_path):
    """Parse a Darshan file in a subprocess to avoid double-free crashes."""
    script = f"""
import json, sys, os
sys.path.insert(0, "{PROJECT_DIR}")
os.chdir("{PROJECT_DIR}")
from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features
import pandas as pd
from src.data.preprocessing import stage3_engineer

parsed = parse_darshan_log("{darshan_path}")
if parsed is None:
    print(json.dumps({{"error": "parse_darshan_log returned None"}}))
    sys.exit(0)

raw = extract_raw_features(parsed)
df = pd.DataFrame([raw])
df = stage3_engineer(df)
features = df.iloc[0].to_dict()

# Convert numpy types to Python native
clean = {{}}
for k, v in features.items():
    if hasattr(v, 'item'):
        clean[k] = v.item()
    elif isinstance(v, float) and (v != v):  # NaN
        clean[k] = 0.0
    else:
        clean[k] = v

print(json.dumps(clean))
"""
    python_bin = sys.executable
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_DIR),
    )

    if result.returncode != 0:
        logger.error("Subprocess parse failed for %s:\nSTDERR: %s",
                      darshan_path, result.stderr[-2000:])
        return None

    try:
        data = json.loads(result.stdout.strip().split("\n")[-1])
        if "error" in data:
            logger.error("Parse error for %s: %s", darshan_path, data["error"])
            return None
        return data
    except (json.JSONDecodeError, IndexError) as exc:
        logger.error("JSON decode failed for %s: %s\nstdout=%s",
                      darshan_path, exc, result.stdout[-500:])
        return None


def find_darshan_files():
    """Map trace names to .darshan file paths."""
    files = {}
    for p in DARSHAN_DIR.iterdir():
        if p.suffix == ".darshan":
            # Extract the trace key from the filename
            # e.g., "(E2E Baseline) original_jeanbez_write_..." -> "(E2E_Baseline)_original..."
            stem = p.stem
            # The trace_labels.json keys use underscores where filenames have spaces
            key = stem.replace(" ", "_")
            files[key] = p

            # Also store a "friendly name" mapping
            if "(" in stem and ")" in stem:
                friendly = stem.split(")")[0].replace("(", "").strip()
            else:
                friendly = stem[:40]
            files[key + "_friendly"] = friendly

    return files


def match_trace_to_label(file_key, label_keys):
    """Find matching label key for a file key."""
    # Direct match
    if file_key in label_keys:
        return file_key
    # Try prefix match
    for lk in label_keys:
        if file_key.startswith(lk) or lk.startswith(file_key):
            return lk
    # Try filename-based match (the label keys may not have the full filename)
    for lk in label_keys:
        # Get the main part after the prefix like "(E2E_Baseline)_"
        lk_parts = lk.split("_", 1) if ")" in lk.split("_", 1)[0] else [lk]
        fk_parts = file_key.split("_", 1) if ")" in file_key.split("_", 1)[0] else [file_key]
        if lk_parts[0] == fk_parts[0]:
            return lk
    return None


def compute_ml_vs_gt(predictions, detected, gt_dims):
    """Compare ML predictions against ground-truth dimensions."""
    pred_set = set(d for d in detected if d != "healthy")
    gt_set = set(gt_dims)

    tp = sorted(pred_set & gt_set)
    fp = sorted(pred_set - gt_set)
    fn = sorted(gt_set - pred_set)
    tn = sorted(set(OUR_DIMENSIONS) - pred_set - gt_set)

    # Per-dimension results
    per_dim = {}
    for dim in OUR_DIMENSIONS:
        conf = predictions.get(dim, 0)
        gt = 1 if dim in gt_set else 0
        pred = 1 if dim in pred_set else 0
        if gt == 1 and pred == 1:
            status = "TP"
        elif gt == 0 and pred == 0:
            status = "TN"
        elif gt == 0 and pred == 1:
            status = "FP"
        else:
            status = "FN"
        per_dim[dim] = {
            "confidence": conf,
            "ground_truth": gt,
            "predicted": pred,
            "status": status,
        }

    n_tp = len(tp)
    n_fp = len(fp)
    n_fn = len(fn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 1.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_dimension": per_dim,
    }


def check_nc_nofill_recommendation(recommendation):
    """Check if LLM recommends NC_NOFILL (the known correct fix from Bez et al.)."""
    if not recommendation:
        return False
    text = json.dumps(recommendation).lower()
    keywords = ["nc_nofill", "nofill", "nc_fill", "fill mode", "nf90_nofill",
                "set_fill", "def_var_fill", "nc_set_fill"]
    return any(kw in text for kw in keywords)


def main():
    logger.info("=" * 70)
    logger.info("TraceBench Real Application Pipeline Evaluation (Tier 1 E2E)")
    logger.info("=" * 70)

    # Load TraceBench labels
    gt_labels = load_tracebench_labels()
    logger.info("Loaded ground truth for %d traces", len(gt_labels))
    for tk, tv in gt_labels.items():
        logger.info("  %s: %s -> %s", tk, tv["tracebench_labels"], tv["mapped_dims"])

    # Find Darshan files
    all_files = {}
    friendly_names = {}
    for p in sorted(DARSHAN_DIR.iterdir()):
        if p.suffix == ".darshan":
            key = p.stem.replace(" ", "_")
            all_files[key] = p
            if "(" in p.stem and ")" in p.stem:
                friendly = p.stem.split(")")[0].replace("(", "").strip()
            else:
                friendly = p.stem[:40]
            friendly_names[key] = friendly

    logger.info("Found %d Darshan files", len(all_files))

    # Match files to labels
    matched = {}
    for fk, fp in all_files.items():
        lk = match_trace_to_label(fk, list(gt_labels.keys()))
        if lk:
            matched[fk] = {"file": fp, "label_key": lk, "gt": gt_labels[lk]}
            logger.info("  Matched: %s -> %s", friendly_names.get(fk, fk), lk)
        else:
            logger.warning("  No label match for: %s", fk)

    # Initialize pipeline (outside parsing loop)
    logger.info("")
    logger.info("Initializing IOPrescriber pipeline...")
    from src.ioprescriber.pipeline import IOPrescriber
    pipeline = IOPrescriber(llm_model="claude-sonnet")

    # Process each trace
    all_results = []
    aggregate = {
        "total_tp": 0, "total_fp": 0, "total_fn": 0,
        "traces_parsed": 0, "traces_failed": 0,
        "nc_nofill_checked": False, "nc_nofill_recommended": False,
    }

    for fk in sorted(matched.keys()):
        info = matched[fk]
        darshan_path = info["file"]
        gt = info["gt"]
        friendly = friendly_names.get(fk, fk)

        logger.info("")
        logger.info("=" * 70)
        logger.info("TRACE: %s", friendly)
        logger.info("  File: %s", darshan_path.name)
        logger.info("  TraceBench labels: %s", gt["tracebench_labels"])
        logger.info("  Mapped dims: %s", gt["mapped_dims"])
        logger.info("=" * 70)

        result = {
            "trace_name": friendly,
            "file_key": fk,
            "file_name": darshan_path.name,
            "tracebench_labels": gt["tracebench_labels"],
            "mapped_ground_truth": gt["mapped_dims"],
            "mapping_details": gt["mapping_details"],
        }

        # Step 0: Parse Darshan (in subprocess)
        logger.info("Step 0: Parsing Darshan log (subprocess)...")
        t0 = time.perf_counter()
        features = parse_darshan_subprocess(str(darshan_path))
        parse_time = time.perf_counter() - t0

        if features is None:
            logger.error("FAILED to parse %s (%.1fs)", friendly, parse_time)
            result["status"] = "PARSE_FAILED"
            result["parse_time_s"] = round(parse_time, 1)
            all_results.append(result)
            aggregate["traces_failed"] += 1
            continue

        logger.info("  Parsed in %.1fs, %d features", parse_time, len(features))
        result["parse_time_s"] = round(parse_time, 1)
        result["n_features"] = len(features)
        aggregate["traces_parsed"] += 1

        # Job summary
        result["job_summary"] = {
            "nprocs": features.get("nprocs", 0),
            "runtime_seconds": round(float(features.get("runtime_seconds", 0)), 1),
            "POSIX_BYTES_WRITTEN": float(features.get("POSIX_BYTES_WRITTEN", 0)),
            "POSIX_BYTES_READ": float(features.get("POSIX_BYTES_READ", 0)),
            "avg_write_size": round(float(features.get("avg_write_size", 0)), 1),
            "avg_read_size": round(float(features.get("avg_read_size", 0)), 1),
            "small_io_ratio": round(float(features.get("small_io_ratio", 0)), 4),
            "seq_write_ratio": round(float(features.get("seq_write_ratio", 0)), 4),
            "seq_read_ratio": round(float(features.get("seq_read_ratio", 0)), 4),
            "metadata_time_ratio": round(float(features.get("metadata_time_ratio", 0)), 4),
            "collective_ratio": round(float(features.get("collective_ratio", 0)), 4),
            "total_bw_mb_s": round(float(features.get("total_bw_mb_s", 0)), 2),
            "has_mpiio": int(features.get("has_mpiio", 0)),
        }

        # Run pipeline
        logger.info("Running IOPrescriber pipeline...")
        try:
            pipe_result = pipeline.analyze(features, workload_name=friendly)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", friendly, exc)
            traceback.print_exc()
            result["status"] = "PIPELINE_FAILED"
            result["error"] = str(exc)
            all_results.append(result)
            continue

        # Extract pipeline outputs
        predictions = pipe_result["step1_detection"]["predictions"]
        detected = pipe_result["step1_detection"]["detected"]

        result["ml_predictions"] = predictions
        result["ml_detected"] = detected

        # ML vs GT comparison
        ml_eval = compute_ml_vs_gt(predictions, detected, gt["mapped_dims"])
        result["ml_evaluation"] = ml_eval
        aggregate["total_tp"] += ml_eval["n_tp"]
        aggregate["total_fp"] += ml_eval["n_fp"]
        aggregate["total_fn"] += ml_eval["n_fn"]

        logger.info("  ML: TP=%s, FP=%s, FN=%s (P=%.2f R=%.2f F1=%.2f)",
                     ml_eval["true_positives"], ml_eval["false_positives"],
                     ml_eval["false_negatives"], ml_eval["precision"],
                     ml_eval["recall"], ml_eval["f1"])

        # SHAP
        result["shap_top_features"] = {}
        for dim, feats in pipe_result.get("step2_shap", {}).items():
            if feats:
                result["shap_top_features"][dim] = [
                    {"feature": f["feature"], "importance": round(f["abs_importance"], 4)}
                    for f in feats[:3]
                ]

        # KB retrieval
        result["kb_retrieval"] = pipe_result.get("step3_retrieval", {})

        # LLM recommendation
        rec = pipe_result.get("step4_recommendation", {})
        result["llm_recommendation"] = {
            "parsed": rec.get("parsed"),
            "groundedness": rec.get("groundedness"),
            "metadata": rec.get("metadata"),
        }

        # Check for NC_NOFILL (E2E traces)
        is_e2e = "E2E" in friendly or "e2e" in friendly
        if is_e2e:
            nofill = check_nc_nofill_recommendation(rec.get("parsed"))
            result["nc_nofill_check"] = {
                "is_e2e_trace": True,
                "nc_nofill_recommended": nofill,
                "note": "Bez et al. ISC 2023 showed NC_NOFILL is the correct fix for E2E write bottleneck",
            }
            if "Baseline" in friendly:
                aggregate["nc_nofill_checked"] = True
                aggregate["nc_nofill_recommended"] = nofill

        result["pipeline_latency_ms"] = pipe_result.get("pipeline_latency_ms", 0)
        result["status"] = "SUCCESS"
        all_results.append(result)

    # Compute aggregate metrics
    tp = aggregate["total_tp"]
    fp = aggregate["total_fp"]
    fn = aggregate["total_fn"]
    agg_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    agg_rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0

    aggregate["precision"] = round(agg_prec, 4)
    aggregate["recall"] = round(agg_rec, 4)
    aggregate["f1"] = round(agg_f1, 4)

    # Groundedness aggregate
    gs_total = 0
    gs_grounded = 0
    for r in all_results:
        gs = r.get("llm_recommendation", {}).get("groundedness")
        if gs:
            gs_total += gs.get("n_recommendations", 0)
            gs_grounded += gs.get("n_grounded", 0)
    aggregate["groundedness_total"] = gs_total
    aggregate["groundedness_grounded"] = gs_grounded
    aggregate["groundedness_score"] = round(gs_grounded / gs_total, 4) if gs_total > 0 else 0

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "tracebench_real_app_results.json"
    final = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "darshan_dir": str(DARSHAN_DIR),
            "n_traces": len(all_results),
            "n_parsed": aggregate["traces_parsed"],
            "n_failed": aggregate["traces_failed"],
        },
        "aggregate_metrics": aggregate,
        "per_trace_results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("")
    logger.info("Results saved: %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("TRACEBENCH REAL APPLICATION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nTraces parsed: {aggregate['traces_parsed']}/{len(matched)}")
    print(f"Traces failed: {aggregate['traces_failed']}")
    print(f"\nAggregate ML Detection (vs TraceBench labels):")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision:       {agg_prec:.4f}")
    print(f"  Recall:          {agg_rec:.4f}")
    print(f"  F1:              {agg_f1:.4f}")
    print(f"\nGroundedness: {gs_grounded}/{gs_total} = {aggregate['groundedness_score']:.4f}")

    if aggregate["nc_nofill_checked"]:
        print(f"\nNC_NOFILL recommended for E2E Baseline: {aggregate['nc_nofill_recommended']}")

    print("\nPer-trace results:")
    for r in all_results:
        status = r.get("status", "UNKNOWN")
        name = r["trace_name"]
        if status == "PARSE_FAILED":
            print(f"\n  {name}: PARSE FAILED")
            continue
        if status == "PIPELINE_FAILED":
            print(f"\n  {name}: PIPELINE FAILED ({r.get('error', '?')})")
            continue

        ev = r.get("ml_evaluation", {})
        print(f"\n  {name}:")
        print(f"    GT labels: {r['tracebench_labels']}")
        print(f"    GT dims:   {r['mapped_ground_truth']}")
        print(f"    ML:        {r['ml_detected']}")
        print(f"    TP={ev.get('true_positives', [])}, FP={ev.get('false_positives', [])}, FN={ev.get('false_negatives', [])}")
        print(f"    P={ev.get('precision', 0):.2f} R={ev.get('recall', 0):.2f} F1={ev.get('f1', 0):.2f}")

        # Top SHAP features
        shap = r.get("shap_top_features", {})
        for dim, feats in shap.items():
            if feats:
                top = feats[0]
                print(f"    SHAP {dim}: {top['feature']} ({top['importance']:.3f})")

        # LLM summary
        rec = r.get("llm_recommendation", {})
        parsed = rec.get("parsed")
        gs = rec.get("groundedness", {})
        if parsed:
            n_recs = len(parsed.get("recommendations", []))
            score = gs.get("groundedness_score", 0) if gs else 0
            print(f"    LLM: {n_recs} recs, groundedness={score:.2f}")
            for i, rx in enumerate(parsed.get("recommendations", [])[:3]):
                dim_name = rx.get("bottleneck_dimension", "?")
                expl = rx.get("explanation", "N/A")[:80]
                print(f"      {i+1}. [{dim_name}] {expl}")
        else:
            print(f"    LLM: No parsed output")

        # NC_NOFILL check
        nc = r.get("nc_nofill_check")
        if nc:
            print(f"    NC_NOFILL recommended: {nc['nc_nofill_recommended']}")

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
