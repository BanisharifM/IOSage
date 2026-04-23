"""
Comprehensive TraceBench evaluation: ALL 35 traces, 3-way comparison, bootstrap CIs.

Runs IOSage full pipeline on all 35 TraceBench traces (real_app_bench, IO500,
single_issue_bench) and compares against IONavigator 1.0 and Drishti outputs.

This is the DEFINITIVE TraceBench evaluation for the IOSage paper.

Usage:
    source .env && python scripts/run_tracebench_full_evaluation.py [--ml-only] [--subset all|real_app|io500|single]

Outputs:
    results/e2e_evaluation/tracebench_full_evaluation.json
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tracebench_full")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACEBENCH_ROOT = (
    PROJECT_DIR / "data" / "external" / "tracebench" / "TraceBench"
)
DATASETS_DIR = TRACEBENCH_ROOT / "Datasets"
LLMEVAL_DIR = TRACEBENCH_ROOT / "LLMEval"
LABEL_MAPPING_FILE = (
    PROJECT_DIR / "data" / "external" / "tracebench" / "label_mapping.json"
)
OUTPUT_DIR = PROJECT_DIR / "results" / "e2e_evaluation"

SUBSETS = ["real_app_bench", "single_issue_bench", "IO500"]

OUR_DIMENSIONS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
    "throughput_utilization",
]

# Dimensions TraceBench actually labels (excludes throughput_utilization)
TRACEBENCH_DIMS = [
    "access_granularity",
    "metadata_intensity",
    "parallelism_efficiency",
    "access_pattern",
    "interface_choice",
    "file_strategy",
]


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------
def load_label_mapping():
    """Load TraceBench 16-label to our 8-dimension mapping."""
    with open(LABEL_MAPPING_FILE) as f:
        mapping = json.load(f)
    tb_to_dim = {}
    for tb_label, info in mapping["tracebench_to_our_taxonomy"].items():
        dim = info["our_dimension"]
        if dim is not None:
            tb_to_dim[tb_label] = dim
    return tb_to_dim


def load_all_traces():
    """Load ground truth labels for ALL TraceBench traces across all subsets."""
    tb_to_dim = load_label_mapping()
    all_traces = {}

    for subset in SUBSETS:
        label_file = DATASETS_DIR / subset / "trace_labels.json"
        if not label_file.exists():
            logger.warning("No trace_labels.json for subset %s", subset)
            continue

        with open(label_file) as f:
            raw_labels = json.load(f)

        darshan_dir = DATASETS_DIR / subset / "darshan_files" / "darshan"

        for trace_key, tb_labels in raw_labels.items():
            # Map to our dimensions
            dims_active = set()
            for lbl in tb_labels:
                dim = tb_to_dim.get(lbl)
                if dim:
                    dims_active.add(dim)

            # Find matching .darshan file
            darshan_file = find_darshan_file(darshan_dir, trace_key)

            all_traces[trace_key] = {
                "subset": subset,
                "tracebench_labels": tb_labels,
                "mapped_dims": sorted(dims_active),
                "darshan_file": darshan_file,
            }

    return all_traces


def find_darshan_file(darshan_dir, trace_key):
    """Find the .darshan file matching a trace key."""
    if not darshan_dir.exists():
        return None

    # Try exact match (with space→underscore)
    for p in darshan_dir.iterdir():
        if p.suffix == ".darshan":
            key = p.stem.replace(" ", "_")
            if key == trace_key or key.startswith(trace_key) or trace_key.startswith(key):
                return p

    # Prefix match for partial keys
    for p in darshan_dir.iterdir():
        if p.suffix == ".darshan":
            key = p.stem.replace(" ", "_")
            # Check if the first part matches (before first underscore after parenthesis)
            if "(" in trace_key and "(" in key:
                tk_prefix = trace_key.split(")")[0]
                k_prefix = key.split(")")[0]
                if tk_prefix == k_prefix:
                    return p

    return None


# ---------------------------------------------------------------------------
# Darshan parsing (subprocess for safety)
# ---------------------------------------------------------------------------
def parse_darshan_subprocess(darshan_path):
    """Parse a Darshan file in a subprocess to avoid crashes."""
    script = f"""
import json, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "{PROJECT_DIR}")
os.chdir("{PROJECT_DIR}")
from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features
import pandas as pd
from src.data.preprocessing import stage3_engineer

try:
    parsed = parse_darshan_log("{darshan_path}")
    if parsed is None:
        print(json.dumps({{"error": "parse_darshan_log returned None"}}))
        sys.exit(0)

    raw = extract_raw_features(parsed)
    df = pd.DataFrame([raw])
    df = stage3_engineer(df)
    features = df.iloc[0].to_dict()

    clean = {{}}
    for k, v in features.items():
        if hasattr(v, 'item'):
            clean[k] = v.item()
        elif isinstance(v, float) and (v != v):
            clean[k] = 0.0
        else:
            clean[k] = v

    print(json.dumps(clean))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=120,
            cwd=str(PROJECT_DIR),
        )
    except subprocess.TimeoutExpired:
        logger.error("Subprocess timed out for %s", darshan_path)
        return None

    try:
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
    except Exception:
        stdout = ""
        stderr = "(decode error)"

    if result.returncode != 0:
        logger.error("Subprocess failed for %s:\n%s", darshan_path, stderr[-1000:])
        return None

    try:
        data = json.loads(stdout.strip().split("\n")[-1])
        if "error" in data:
            logger.error("Parse error for %s: %s", darshan_path, data["error"])
            return None
        return data
    except (json.JSONDecodeError, IndexError) as exc:
        logger.error("JSON decode failed for %s: %s", darshan_path, exc)
        return None


# ---------------------------------------------------------------------------
# IONavigator output parsing (from tracebench_comprehensive_eval.py)
# ---------------------------------------------------------------------------
NEGATION_PATTERN = re.compile(
    r"(no\s+(significant|major|critical|notable|apparent|clear)\s+(issue|problem|concern|bottleneck))"
    r"|(not\s+(a|an)\s+(significant|major|critical)\s+(issue|concern|bottleneck))"
    r"|(negligible|minimal\s+impact|unlikely\s+to\s+be|not\s+a\s+widespread|ineffective)"
    r"|(no\s+issue|no\s+problems?\s+detected|no\s+concerns?)"
    r"|(does\s+not\s+(appear|seem)\s+to\s+be\s+(a|an)\s+(issue|problem))",
    re.IGNORECASE,
)

POSITIVE_PATTERN = re.compile(
    r"(significant|issue|concern|impact|problem|bottleneck|degradation|inefficien|hindering|overhead)",
    re.IGNORECASE,
)

ION_SECTION_KEYWORDS = {
    "access_granularity": [r"small\s+(i/o|io|read|write)", r"misalign"],
    "access_pattern": [r"random\s+(i/o|io|access|read|write)", r"non.?sequential"],
    "interface_choice": [r"non.?collective", r"collective\s+i/o", r"low.?level\s+library",
                          r"without\s+mpi", r"no\s+collective", r"posix\s+instead"],
    "parallelism_efficiency": [r"load\s+imbalanc", r"server\s+load", r"rank\s+load",
                                r"imbalance"],
    "metadata_intensity": [r"metadata\s+(i/o|io|load|overhead|time|operations?)",
                            r"high\s+metadata"],
    "file_strategy": [r"shared\s+file", r"file.?per.?process", r"contention",
                       r"lock\s+contention"],
    "throughput_utilization": [r"low\s+throughput", r"bandwidth", r"underutiliz",
                                r"low\s+bandwidth", r"peak\s+performance"],
}


def parse_ion_detections(ion_text):
    """Parse IONavigator 1.0 diagnosis text into detected dimensions."""
    if not ion_text:
        return set()

    detected = set()
    paragraphs = re.split(r"\n\s*\n|\n(?=[A-Z])", ion_text)

    for para in paragraphs:
        para_lower = para.lower()
        if len(para.strip()) < 20:
            continue

        # Check negation
        if NEGATION_PATTERN.search(para):
            has_positive = POSITIVE_PATTERN.search(para)
            if not has_positive:
                continue

        for dim, patterns in ION_SECTION_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, para_lower):
                    # Verify positive context
                    if POSITIVE_PATTERN.search(para):
                        detected.add(dim)
                    break

    return detected


def parse_drishti_detections(drishti_text):
    """Parse Drishti output into detected dimensions."""
    if not drishti_text:
        return set()

    detected = set()
    text_lower = drishti_text.lower()

    drishti_patterns = {
        "access_granularity": [
            r"small\s+(read|write)\s+requests",
            r"misaligned\s+file\s+requests",
            r"misaligned\s+memory",
        ],
        "parallelism_efficiency": [
            r"load\s+imbalance.*detected",
            r"data\s+sieving",
        ],
        "interface_choice": [
            r"does\s+not\s+use\s+collective\s+(read|write)",
            r"no\s+collective",
        ],
        "file_strategy": [
            r"small\s+(read|write)\s+requests\s+to\s+a\s+shared\s+file",
            r"shared\s+file.*contention",
        ],
        "metadata_intensity": [
            r"high\s+metadata",
            r"metadata.*time.*high",
        ],
        "access_pattern": [
            r"random\s+(read|write)\s+operations",
            r"random\s+access",
        ],
        "throughput_utilization": [
            r"redundant\s+(read|write)\s+traffic",
        ],
    }

    for dim, patterns in drishti_patterns.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                detected.add(dim)
                break

    return detected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred_set, gt_set, eval_dims=None):
    """Compute classification metrics between predicted and ground-truth sets."""
    if eval_dims is None:
        eval_dims = TRACEBENCH_DIMS

    pred_filtered = {d for d in pred_set if d in eval_dims}
    gt_filtered = {d for d in gt_set if d in eval_dims}

    tp = sorted(pred_filtered & gt_filtered)
    fp = sorted(pred_filtered - gt_filtered)
    fn = sorted(gt_filtered - pred_filtered)

    n_tp = len(tp)
    n_fp = len(fp)
    n_fn = len(fn)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 1.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bootstrap_ci(trace_results, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute BCa bootstrap confidence interval for aggregate F1."""
    rng = np.random.RandomState(seed)
    n = len(trace_results)
    if n < 3:
        return {"ci_lower": 0, "ci_upper": 1, "n": n, "note": "too few samples"}

    def compute_f1(indices):
        tp = sum(trace_results[i]["n_tp"] for i in indices)
        fp = sum(trace_results[i]["n_fp"] for i in indices)
        fn = sum(trace_results[i]["n_fn"] for i in indices)
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    # Original statistic
    orig_f1 = compute_f1(list(range(n)))

    # Bootstrap resamples
    boot_f1s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_f1s.append(compute_f1(idx))

    boot_f1s = np.array(boot_f1s)
    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_f1s, alpha * 100))
    ci_upper = float(np.percentile(boot_f1s, (1 - alpha) * 100))

    return {
        "f1": round(orig_f1, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_traces": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full TraceBench evaluation")
    parser.add_argument("--ml-only", action="store_true",
                        help="ML detection only (no SHAP/KB/LLM)")
    parser.add_argument("--subset", default="all",
                        choices=["all", "real_app_bench", "IO500", "single_issue_bench"],
                        help="Which subset to evaluate")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE TRACEBENCH EVALUATION (ALL 35 TRACES)")
    logger.info("=" * 70)

    # Load all traces
    all_traces = load_all_traces()
    if args.subset != "all":
        all_traces = {k: v for k, v in all_traces.items() if v["subset"] == args.subset}
    logger.info("Loaded %d traces across %d subsets", len(all_traces),
                 len(set(v["subset"] for v in all_traces.values())))

    # Initialize pipeline (if not ML-only)
    pipeline = None
    if not args.ml_only:
        logger.info("Initializing IOSage pipeline...")
        try:
            from src.ioprescriber.pipeline import IOPrescriber
            pipeline = IOPrescriber(llm_model="claude-sonnet")
        except Exception as exc:
            logger.warning("Pipeline init failed (%s), falling back to ML-only", exc)
            args.ml_only = True

    # Load ML models
    logger.info("Loading ML models...")
    import pickle
    model_path = PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl"
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    scalers_path = PROJECT_DIR / "data" / "processed" / "production" / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    import yaml
    config_path = PROJECT_DIR / "configs" / "preprocessing.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Process each trace
    per_trace_results = []
    systems = {"iosage": [], "ionavigator": [], "drishti": []}

    for trace_key in sorted(all_traces.keys()):
        trace = all_traces[trace_key]
        subset = trace["subset"]
        gt_dims = set(trace["mapped_dims"])

        # Friendly name
        if "(" in trace_key and ")" in trace_key:
            friendly = trace_key.split(")")[0].replace("(", "").strip()
        else:
            friendly = trace_key[:50]

        logger.info("")
        logger.info("--- %s [%s] ---", friendly, subset)
        logger.info("  GT labels: %s -> dims: %s", trace["tracebench_labels"], trace["mapped_dims"])

        result = {
            "trace_key": trace_key,
            "friendly_name": friendly,
            "subset": subset,
            "tracebench_labels": trace["tracebench_labels"],
            "mapped_ground_truth": trace["mapped_dims"],
        }

        # 1. IOSage ML detection
        darshan_file = trace["darshan_file"]
        if darshan_file is None:
            logger.warning("  No .darshan file found for %s", trace_key)
            result["iosage_status"] = "NO_DARSHAN_FILE"
            result["iosage_metrics"] = None
        else:
            features = parse_darshan_subprocess(str(darshan_file))
            if features is None:
                logger.warning("  Parse failed for %s", trace_key)
                result["iosage_status"] = "PARSE_FAILED"
                result["iosage_metrics"] = None
            else:
                if pipeline and not args.ml_only:
                    try:
                        pipe_result = pipeline.analyze(features, workload_name=friendly)
                        detected = set(pipe_result["step1_detection"]["detected"])
                        detected.discard("healthy")
                        result["iosage_status"] = "SUCCESS"
                        result["iosage_detected"] = sorted(detected)
                        result["iosage_predictions"] = pipe_result["step1_detection"]["predictions"]
                    except Exception as exc:
                        logger.error("  Pipeline failed: %s", exc)
                        result["iosage_status"] = "PIPELINE_FAILED"
                        result["iosage_metrics"] = None
                        detected = set()
                else:
                    # ML-only: replicate Detector.detect_from_features()
                    # The Detector uses RAW engineered features (NO stage5 normalization)
                    # XGBoost is tree-based → invariant to monotone transforms
                    import pandas as pd

                    # Get feature columns same way as Detector class
                    if not hasattr(main, '_feature_cols'):
                        import yaml as _yaml
                        cfg_path = PROJECT_DIR / "configs" / "training.yaml"
                        with open(cfg_path) as _f:
                            _cfg = _yaml.safe_load(_f)
                        prod_feat = pd.read_parquet(
                            PROJECT_DIR / _cfg["paths"]["production_features"]
                        )
                        exclude = set(_cfg.get("exclude_features", []))
                        for col in prod_feat.columns:
                            if col.startswith("_") or col.startswith("drishti_"):
                                exclude.add(col)
                        main._feature_cols = [c for c in prod_feat.columns if c not in exclude]

                    feature_cols = main._feature_cols
                    X = np.array([[features.get(col, 0) for col in feature_cols]],
                                  dtype=np.float32)

                    detected = set()
                    predictions = {}
                    models_dict = model_bundle if isinstance(model_bundle, dict) else model_bundle.get("models", model_bundle)
                    for dim in OUR_DIMENSIONS:
                        model = models_dict.get(dim)
                        if model is None:
                            predictions[dim] = 0.0
                            continue
                        try:
                            prob = float(model.predict_proba(X)[0][1])
                        except Exception as e:
                            logger.debug("Prediction failed for %s: %s", dim, e)
                            prob = 0.0
                        predictions[dim] = round(prob, 4)
                        if prob >= 0.3:  # Same threshold as IOPrescriber Detector
                            detected.add(dim)

                    result["iosage_status"] = "SUCCESS"
                    result["iosage_detected"] = sorted(detected)
                    result["iosage_predictions"] = predictions

                if result.get("iosage_status") == "SUCCESS":
                    m = compute_metrics(detected, gt_dims)
                    result["iosage_metrics"] = m
                    systems["iosage"].append(m)
                    logger.info("  IOSage: TP=%s FP=%s FN=%s (F1=%.3f)",
                                 m["tp"], m["fp"], m["fn"], m["f1"])

        # 2. IONavigator 1.0 detection (from pre-existing outputs)
        ion_file = LLMEVAL_DIR / trace_key / "ION-1.0_diagnosis.txt"
        if ion_file.exists():
            ion_text = ion_file.read_text(errors="replace")
            ion_detected = parse_ion_detections(ion_text)
            m = compute_metrics(ion_detected, gt_dims)
            result["ionavigator_detected"] = sorted(ion_detected)
            result["ionavigator_metrics"] = m
            systems["ionavigator"].append(m)
            logger.info("  IONav:  TP=%s FP=%s FN=%s (F1=%.3f)",
                         m["tp"], m["fp"], m["fn"], m["f1"])
        else:
            result["ionavigator_detected"] = None
            result["ionavigator_metrics"] = None
            logger.info("  IONav: no diagnosis file")

        # 3. Drishti detection (from pre-existing outputs)
        drishti_file = LLMEVAL_DIR / trace_key / "drishti.txt"
        if drishti_file.exists():
            drishti_text = drishti_file.read_text(errors="replace")
            drishti_detected = parse_drishti_detections(drishti_text)
            m = compute_metrics(drishti_detected, gt_dims)
            result["drishti_detected"] = sorted(drishti_detected)
            result["drishti_metrics"] = m
            systems["drishti"].append(m)
            logger.info("  Drishti: TP=%s FP=%s FN=%s (F1=%.3f)",
                         m["tp"], m["fp"], m["fn"], m["f1"])
        else:
            result["drishti_detected"] = None
            result["drishti_metrics"] = None
            logger.info("  Drishti: no output file")

        per_trace_results.append(result)

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 70)

    aggregates = {}
    for sys_name, trace_metrics in systems.items():
        if not trace_metrics:
            continue
        total_tp = sum(m["n_tp"] for m in trace_metrics)
        total_fp = sum(m["n_fp"] for m in trace_metrics)
        total_fn = sum(m["n_fn"] for m in trace_metrics)

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Bootstrap CI
        ci = bootstrap_ci(trace_metrics)

        aggregates[sys_name] = {
            "n_traces": len(trace_metrics),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "bootstrap_ci": ci,
        }

        logger.info("  %s: P=%.3f R=%.3f F1=%.3f [%.3f, %.3f] (n=%d traces)",
                     sys_name, p, r, f1, ci["ci_lower"], ci["ci_upper"],
                     len(trace_metrics))

    # Per-dimension breakdown
    per_dim_results = {}
    for dim in TRACEBENCH_DIMS:
        per_dim_results[dim] = {}
        for sys_name, trace_list in [("iosage", per_trace_results),
                                      ("ionavigator", per_trace_results),
                                      ("drishti", per_trace_results)]:
            tp = fp = fn = 0
            for tr in trace_list:
                gt = set(tr["mapped_ground_truth"])
                det_key = f"{sys_name}_detected"
                det = tr.get(det_key)
                if det is None:
                    continue
                det = set(det)
                if dim in gt and dim in det:
                    tp += 1
                elif dim not in gt and dim in det:
                    fp += 1
                elif dim in gt and dim not in det:
                    fn += 1

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_dim_results[dim][sys_name] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            }

    # Per-subset breakdown
    per_subset_agg = {}
    for subset in SUBSETS:
        per_subset_agg[subset] = {}
        for sys_name in ["iosage", "ionavigator", "drishti"]:
            traces_in_subset = [
                tr for tr in per_trace_results
                if tr["subset"] == subset and tr.get(f"{sys_name}_metrics") is not None
            ]
            if not traces_in_subset:
                continue
            metrics = [tr[f"{sys_name}_metrics"] for tr in traces_in_subset]
            total_tp = sum(m["n_tp"] for m in metrics)
            total_fp = sum(m["n_fp"] for m in metrics)
            total_fn = sum(m["n_fn"] for m in metrics)
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_subset_agg[subset][sys_name] = {
                "n_traces": len(traces_in_subset),
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "tracebench_full_evaluation.json"

    n_parsed = sum(1 for r in per_trace_results if r.get("iosage_status") == "SUCCESS")
    n_failed = sum(1 for r in per_trace_results if r.get("iosage_status") in ("PARSE_FAILED", "NO_DARSHAN_FILE", "PIPELINE_FAILED"))

    final = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_traces_total": len(per_trace_results),
            "n_traces_parsed": n_parsed,
            "n_traces_failed": n_failed,
            "mode": "ml_only" if args.ml_only else "full_pipeline",
            "subset_filter": args.subset,
            "eval_dimensions": TRACEBENCH_DIMS,
            "note": "throughput_utilization excluded from eval (TraceBench does not label it)",
        },
        "aggregate_comparison": aggregates,
        "per_dimension_comparison": per_dim_results,
        "per_subset_comparison": per_subset_agg,
        "per_trace_results": per_trace_results,
    }

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("\nResults saved: %s", output_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("3-WAY COMPARISON: IOSage vs IONavigator vs Drishti")
    print("=" * 70)
    print(f"\nTraces: {len(per_trace_results)} total, {n_parsed} parsed, {n_failed} failed")
    print(f"\n{'System':<15} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'95% CI':>16}")
    print("-" * 70)
    for sys_name in ["iosage", "ionavigator", "drishti"]:
        agg = aggregates.get(sys_name)
        if agg:
            ci = agg["bootstrap_ci"]
            print(f"{sys_name:<15} {agg['precision']:>8.3f} {agg['recall']:>8.3f} "
                  f"{agg['f1']:>8.3f} {agg['total_tp']:>5} {agg['total_fp']:>5} "
                  f"{agg['total_fn']:>5} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    print(f"\nPer-dimension F1:")
    print(f"{'Dimension':<25} {'IOSage':>8} {'IONav':>8} {'Drishti':>8}")
    print("-" * 55)
    for dim in TRACEBENCH_DIMS:
        vals = per_dim_results.get(dim, {})
        print(f"{dim:<25} "
              f"{vals.get('iosage', {}).get('f1', 0):>8.3f} "
              f"{vals.get('ionavigator', {}).get('f1', 0):>8.3f} "
              f"{vals.get('drishti', {}).get('f1', 0):>8.3f}")

    print(f"\nPer-subset breakdown:")
    for subset in SUBSETS:
        sub = per_subset_agg.get(subset, {})
        if sub:
            print(f"\n  {subset}:")
            for sys_name in ["iosage", "ionavigator", "drishti"]:
                s = sub.get(sys_name)
                if s:
                    print(f"    {sys_name:<15}: F1={s['f1']:.3f} (P={s['precision']:.3f} R={s['recall']:.3f}, n={s['n_traces']})")

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
