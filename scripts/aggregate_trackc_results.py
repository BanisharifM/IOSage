"""
Aggregate Iterative iterative optimization results into summary metrics.

Computes per-workload, per-model, and ablation statistics for SC paper.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.llm.iterative_result import load_iterative_results  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "resubmission" / "iterative"


def load_all_results(results_dir=RESULTS_DIR):
    """Load only versioned completed primary runs."""
    return load_iterative_results(results_dir)


def categorize_results(results):
    """Categorize results into sweep, ablation, and smoke test."""
    sweep = []
    ablation = []
    smoke = []

    for r in results:
        src = r.get("_source_file", "")
        if src.startswith("trackc_"):
            smoke.append(r)
        elif r["condition"] == "full":
            sweep.append(r)
        else:
            ablation.append(r)

    return sweep, ablation, smoke


def compute_sweep_metrics(sweep_results):
    """Compute per-workload, per-model metrics from sweep results."""
    # Group by (workload, model)
    groups = defaultdict(list)
    for r in sweep_results:
        key = (r["workload"], r["model"])
        groups[key].append(r)

    metrics = {}
    for (workload, model), runs in sorted(groups.items()):
        speedups = [r["best_speedup"] for r in runs]
        iterations = [r["total_iterations"] for r in runs]
        costs = [r["total_cost_usd"] for r in runs]

        metrics[(workload, model)] = {
            "n_valid": len(runs),
            "n_total": len(runs),
            "mean_speedup": float(np.mean(speedups)),
            "std_speedup": float(np.std(speedups)),
            "median_speedup": float(np.median(speedups)),
            "min_speedup": float(np.min(speedups)),
            "max_speedup": float(np.max(speedups)),
            "geo_mean_speedup": float(np.exp(np.mean(np.log(speedups)))),
            "mean_iterations": float(np.mean(iterations)),
            "mean_cost": float(np.mean(costs)),
            "total_cost": float(np.sum(costs)),
            "statuses": [r["final_status"] for r in runs],
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
            "geo_mean_speedup": float(np.exp(np.mean(np.log(all_speedups)))),
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
        groups[r["condition"]].append(r)

    metrics = {}
    for condition, runs in sorted(groups.items()):
        speedups = [r["best_speedup"] for r in runs]
        metrics[condition] = {
            "n_runs": len(runs),
            "mean_speedup": float(np.mean(speedups)),
            "std_speedup": float(np.std(speedups)),
            "geo_mean_speedup": float(np.exp(np.mean(np.log(speedups)))),
            "speedups": speedups,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "trackc_complete_results.json"
    logger.info("Loading results from %s", results_dir)
    all_results = load_all_results(results_dir)
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
        smoke_summary[r["result_id"]] = {
            "workload": r["workload"],
            "speedup": r["best_speedup"],
            "iterations": r["total_iterations"],
            "status": r["final_status"],
            "cost": r["total_cost_usd"],
        }

    # Build complete results
    complete = {
        "schema_version": 1,
        "source_result_ids": sorted(r["result_id"] for r in all_results),
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
    for result_key, s in sorted(smoke_summary.items()):
        print(f"  {s['workload']:30s}  {s['speedup']:6.2f}x  iters={s['iterations']}  {s['status']:20s}  ${s['cost']:.4f}  {result_key}")

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
    if output_path.exists():
        raise FileExistsError(f"aggregate output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(complete, f, indent=2, default=str)
    logger.info("Complete results saved to %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
