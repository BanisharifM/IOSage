"""
Aggregate Iterative iterative optimization results into summary metrics.

Computes per-workload, per-model, and ablation statistics for SC paper.
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "iterative"


def load_all_results():
    """Load all JSON result files from results/iterative/."""
    all_results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for d in data:
                    d["_source_file"] = f.name
                all_results.extend(data)
            elif isinstance(data, dict):
                data["_source_file"] = f.name
                all_results.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f.name, e)
    return all_results


def categorize_results(results):
    """Categorize results into sweep, ablation, and smoke test."""
    sweep = []
    ablation = []
    smoke = []

    for r in results:
        src = r.get("_source_file", "")
        if src.startswith("sweep_"):
            sweep.append(r)
        elif src.startswith("ablation_"):
            ablation.append(r)
        elif src.startswith("trackc_"):
            smoke.append(r)

    return sweep, ablation, smoke


def compute_sweep_metrics(sweep_results):
    """Compute per-workload, per-model metrics from sweep results."""
    # Group by (workload, model)
    groups = defaultdict(list)
    for r in sweep_results:
        key = (r.get("workload", "?"), r.get("model", "?"))
        groups[key].append(r)

    metrics = {}
    for (workload, model), runs in sorted(groups.items()):
        # Filter out failed baselines
        valid_runs = [r for r in runs if r.get("final_status") not in ("baseline_failed", "not_started")]
        if not valid_runs:
            metrics[(workload, model)] = {
                "n_valid": 0,
                "n_total": len(runs),
                "mean_speedup": 0,
                "std_speedup": 0,
                "mean_iterations": 0,
                "mean_cost": 0,
                "statuses": [r.get("final_status") for r in runs],
            }
            continue

        speedups = [r.get("best_speedup", 1.0) for r in valid_runs]
        iterations = [r.get("total_iterations", 0) for r in valid_runs]
        costs = [r.get("total_cost_usd", 0) for r in valid_runs]

        metrics[(workload, model)] = {
            "n_valid": len(valid_runs),
            "n_total": len(runs),
            "mean_speedup": float(np.mean(speedups)),
            "std_speedup": float(np.std(speedups)),
            "median_speedup": float(np.median(speedups)),
            "min_speedup": float(np.min(speedups)),
            "max_speedup": float(np.max(speedups)),
            "geo_mean_speedup": float(np.exp(np.mean(np.log(np.maximum(speedups, 0.01))))),
            "mean_iterations": float(np.mean(iterations)),
            "mean_cost": float(np.mean(costs)),
            "total_cost": float(np.sum(costs)),
            "statuses": [r.get("final_status") for r in valid_runs],
            "speedups": speedups,
        }

    return metrics


def compute_model_summary(sweep_metrics):
    """Compute per-model aggregate metrics across all workloads."""
    model_data = defaultdict(list)
    for (workload, model), m in sweep_metrics.items():
        if m["n_valid"] > 0:
            model_data[model].append(m)

    summary = {}
    for model, mlist in model_data.items():
        all_speedups = []
        all_iterations = []
        all_costs = []
        for m in mlist:
            all_speedups.extend(m["speedups"])
            all_iterations.append(m["mean_iterations"])
            all_costs.append(m["mean_cost"])

        summary[model] = {
            "n_workloads": len(mlist),
            "geo_mean_speedup": float(np.exp(np.mean(np.log(np.maximum(all_speedups, 0.01))))),
            "mean_speedup": float(np.mean(all_speedups)),
            "std_speedup": float(np.std(all_speedups)),
            "mean_iterations": float(np.mean(all_iterations)),
            "mean_cost_per_run": float(np.mean(all_costs)),
            "total_cost": float(np.sum(all_costs)),
        }

    return summary


def compute_ablation_metrics(ablation_results):
    """Compute ablation condition effects."""
    groups = defaultdict(list)
    for r in ablation_results:
        src = r.get("_source_file", "")
        # Extract ablation type from filename: ablation_<condition>_<workload>.json
        parts = src.replace("ablation_", "").replace(".json", "").split("_", 2)
        if len(parts) >= 2:
            condition = parts[0]
            if parts[0] == "no":
                condition = f"no_{parts[1]}"
            elif parts[0] == "single":
                condition = "single_shot"
        else:
            condition = "unknown"
        groups[condition].append(r)

    metrics = {}
    for condition, runs in sorted(groups.items()):
        valid = [r for r in runs if r.get("final_status") not in ("baseline_failed", "not_started")]
        if not valid:
            continue
        speedups = [r.get("best_speedup", 1.0) for r in valid]
        metrics[condition] = {
            "n_runs": len(valid),
            "mean_speedup": float(np.mean(speedups)),
            "std_speedup": float(np.std(speedups)),
            "geo_mean_speedup": float(np.exp(np.mean(np.log(np.maximum(speedups, 0.01))))),
            "speedups": speedups,
        }

    return metrics


def main():
    logger.info("Loading results from %s", RESULTS_DIR)
    all_results = load_all_results()
    logger.info("Loaded %d total result records", len(all_results))

    sweep, ablation, smoke = categorize_results(all_results)
    logger.info("Categories: %d sweep, %d ablation, %d smoke test", len(sweep), len(ablation), len(smoke))

    # Sweep metrics
    sweep_metrics = compute_sweep_metrics(sweep)
    model_summary = compute_model_summary(sweep_metrics)

    # Ablation metrics
    ablation_metrics = compute_ablation_metrics(ablation)

    # Smoke test summary
    smoke_summary = {}
    for r in smoke:
        w = r.get("workload", "?")
        smoke_summary[w] = {
            "speedup": r.get("best_speedup", 1.0),
            "iterations": r.get("total_iterations", 0),
            "status": r.get("final_status", "?"),
            "cost": r.get("total_cost_usd", 0),
        }

    # Build complete results
    complete = {
        "phase1_smoke_test": smoke_summary,
        "phase2_sweep_per_workload_model": {
            f"{w}_{m}": {k: v for k, v in metrics.items() if k != "speedups"}
            for (w, m), metrics in sweep_metrics.items()
        },
        "phase2_model_summary": model_summary,
        "phase3_ablation": ablation_metrics,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TRACK C COMPLETE RESULTS SUMMARY")
    print("=" * 80)

    print("\n--- Phase 1: Smoke Tests ---")
    for w, s in sorted(smoke_summary.items()):
        print(f"  {w:30s}  {s['speedup']:6.2f}x  iters={s['iterations']}  {s['status']:20s}  ${s['cost']:.4f}")

    print("\n--- Phase 2: Sweep Results (per workload x model) ---")
    for (w, m), s in sorted(sweep_metrics.items()):
        if s["n_valid"] > 0:
            print(f"  {w:25s} {m:15s}  {s['mean_speedup']:6.2f}x +/- {s['std_speedup']:.2f}  "
                  f"(geo={s['geo_mean_speedup']:.2f}x)  iters={s['mean_iterations']:.1f}  "
                  f"${s['mean_cost']:.4f}/run  n={s['n_valid']}/{s['n_total']}")

    print("\n--- Phase 2: Model Summary ---")
    for model, s in sorted(model_summary.items()):
        print(f"  {model:15s}  geo_mean={s['geo_mean_speedup']:.2f}x  "
              f"mean={s['mean_speedup']:.2f}x +/- {s['std_speedup']:.2f}  "
              f"iters={s['mean_iterations']:.1f}  ${s['mean_cost_per_run']:.4f}/run  "
              f"workloads={s['n_workloads']}")

    if ablation_metrics:
        print("\n--- Phase 3: Ablation Results ---")
        for cond, s in sorted(ablation_metrics.items()):
            print(f"  {cond:20s}  {s['mean_speedup']:6.2f}x +/- {s['std_speedup']:.2f}  "
                  f"(geo={s['geo_mean_speedup']:.2f}x)  n={s['n_runs']}")

    print("=" * 80)

    # Save
    output_path = RESULTS_DIR / "trackc_complete_results.json"
    with open(output_path, "w") as f:
        json.dump(complete, f, indent=2, default=str)
    logger.info("Complete results saved to %s", output_path)

    return complete


if __name__ == "__main__":
    main()
