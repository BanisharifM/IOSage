#!/usr/bin/env python3
"""
Merge 28 chunked IOAgent GPT-4o results into one full 488-trace metric file,
then compare against the other 5 systems already in final_metrics.json.

Usage:
    python merge_full488_results.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Reuse compute_metrics from the run script (no duplicated logic)
sys.path.insert(0, str(Path(__file__).parent))
from run_ioagent_full488 import compute_metrics, LABEL_DIMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path('/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/ioagent_gpt4o_full488')
PROJECT = Path('/work/hdd/bdau/mbanisharifdehkordi/SC_2026')

CHUNKS_PER_BENCH = {
    'ior': 8, 'mdtest': 4, 'h5bench': 4,
    'dlio': 4, 'hacc_io': 4, 'custom': 4,
}


def merge_ioagent_chunks():
    """Combine 28 raw_results.json files into one 488-trace IOAgent result."""
    all_results = []
    total_api_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    total_elapsed = 0.0
    n_attempted = 0
    missing_chunks = []

    for bench, n_chunks in CHUNKS_PER_BENCH.items():
        for i in range(n_chunks):
            sub = BASE / f'output_{bench}_c{i}'
            raw = sub / 'raw_results.json'
            if not raw.exists():
                missing_chunks.append(f'{bench}_c{i}')
                continue
            with open(raw) as f:
                results = json.load(f)
            all_results.extend(results)
            logger.info(f'  [{bench}_c{i}] {len(results)} traces')

            metrics_path = sub / 'evaluation_metrics.json'
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                api = m.get('api_metrics', {})
                total_api_calls += api.get('total_api_calls', 0)
                total_prompt_tokens += api.get('total_prompt_tokens', 0)
                total_completion_tokens += api.get('total_completion_tokens', 0)
                total_cost_usd += api.get('total_cost_usd', 0.0)
                n_attempted += m.get('n_traces_attempted', len(results))
                avg = m.get('avg_time_per_trace', 0)
                total_elapsed += avg * len(results)

    if missing_chunks:
        logger.warning(f"Missing chunks: {missing_chunks}")

    if not all_results:
        logger.error("No results found")
        return None

    # Build results DataFrame in the shape compute_metrics expects
    rows = []
    for r in all_results:
        row = {'job_id': r['job_id'], 'benchmark': r['benchmark']}
        for dim in LABEL_DIMS:
            row[f'gt_{dim}'] = r['ground_truth'].get(dim, 0)
            row[f'pred_{dim}'] = r['detections'].get(dim, 0)
        row['elapsed_seconds'] = r.get('elapsed_seconds', 0)
        rows.append(row)
    results_df = pd.DataFrame(rows)

    metrics = compute_metrics(results_df, LABEL_DIMS)
    metrics['api_metrics'] = {
        'total_api_calls': total_api_calls,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_cost_usd': round(total_cost_usd, 4),
    }
    metrics['n_traces_processed'] = len(all_results)
    metrics['n_traces_attempted'] = n_attempted
    metrics['model'] = 'openai/gpt-4o'
    metrics['avg_time_per_trace'] = round(total_elapsed / max(len(all_results), 1), 2)
    metrics['per_benchmark_count'] = {
        b: int((results_df['benchmark'] == b).sum()) for b in CHUNKS_PER_BENCH
    }

    # Save merged outputs
    out_dir = BASE / 'merged_full488'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / 'raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    results_df.to_csv(out_dir / 'predictions.csv', index=False)

    logger.info(
        f"Merged {len(all_results)} traces -> "
        f"Micro-F1={metrics['overall']['micro_f1']:.4f}, "
        f"Macro-F1={metrics['overall']['macro_f1']:.4f}, "
        f"cost=${total_cost_usd:.2f}"
    )
    return metrics


def load_new_wisio():
    """Load the freshly-run WisIO results on full 488 (replaces old n=438).
    Returns all 4 metrics needed for tab_baselines: Mi-F1, Ma-F1, Hamming, Subset Acc."""
    new_wisio = PROJECT / 'results/wisio_baseline_full488/wisio_metrics.json'
    if not new_wisio.exists():
        return None
    with open(new_wisio) as f:
        d = json.load(f)
    overall = d.get('overall', {})
    return {
        'micro_f1': overall.get('micro_f1', 0),
        'macro_f1': overall.get('macro_f1', 0),
        'hamming_loss': overall.get('hamming_loss', 0),
        'subset_accuracy': overall.get('subset_accuracy', 0),
        'n_samples': overall.get('n_samples', 0),
    }


def compare_with_other_5():
    """Print Table II-style comparison: IOAgent (488) vs the other 5 systems."""
    final_path = PROJECT / 'results/boost_experiment/full_evaluation/final_metrics.json'
    with open(final_path) as f:
        d = json.load(f)

    # IOAgent merged result
    ioagent_path = BASE / 'merged_full488' / 'evaluation_metrics.json'
    if ioagent_path.exists():
        with open(ioagent_path) as f:
            ioagent = json.load(f)
        ioagent_mi = ioagent['overall']['micro_f1']
        ioagent_ma = ioagent['overall']['macro_f1']
        ioagent_ham = ioagent['overall'].get('hamming_loss', 0)
        ioagent_sa = ioagent['overall'].get('subset_accuracy', 0)
        ioagent_n = ioagent['n_traces_processed']
        ioagent_cost = ioagent['api_metrics']['total_cost_usd']
    else:
        ioagent_mi = ioagent_ma = ioagent_ham = ioagent_sa = ioagent_n = None
        ioagent_cost = 0

    # Fresh WisIO on full 488 (preferred over old n=438 in final_metrics.json)
    new_wisio = load_new_wisio()

    print()
    print("=" * 95)
    print("TABLE II — All systems on the full 488-sample test set")
    print("=" * 95)
    print(f"{'System':<22} {'Type':<10} {'Mi-F1':>8} {'Ma-F1':>8} {'Ham.':>8} {'Sub.A':>8} {'n':>5}  {'Source':<15}")
    print('-' * 95)
    sysmap_pre = [
        ('xgboost',  'IOSage (ours)', 'Pipeline'),
        ('drishti',  'Drishti',       'Rules'),
    ]
    sysmap_post = [
        ('threshold_90pct', 'Threshold (P90)', 'Stat.'),
        ('majority_class',  'Majority class',  'Triv.'),
    ]

    def _row(name, type_, mi, ma, ha, sa, n, src):
        print(f"{name:<22} {type_:<10} {mi:>8.4f} {ma:>8.4f} "
              f"{ha:>8.4f} {sa:>8.4f} {n:>5}  {src}")

    # IOSage and Drishti
    for key, name, type_ in sysmap_pre:
        if key in d:
            m = d[key].get('metrics', d[key])
            _row(name, type_,
                 m.get('micro_f1', 0), m.get('macro_f1', 0),
                 m.get('hamming_loss', 0), m.get('subset_accuracy', 0),
                 d[key].get('n_samples_evaluated', 488), 'final_metrics')

    # IOAgent (now includes hamming/subset_accuracy)
    if ioagent_mi is not None:
        _row('IOAgent (GPT-4o)', 'LLM', ioagent_mi, ioagent_ma,
             ioagent_ham, ioagent_sa, ioagent_n, 'full488 (new)')
    else:
        print(f"{'IOAgent (GPT-4o)':<22} {'LLM':<10} {'PENDING':>8} {'':>8} {'':>8} {'':>8} {'':>5}  full488 (new)")

    # WisIO (prefer new, fall back to old)
    if new_wisio is not None:
        _row('WisIO', 'Rules',
             new_wisio['micro_f1'], new_wisio['macro_f1'],
             new_wisio['hamming_loss'], new_wisio['subset_accuracy'],
             new_wisio['n_samples'], 'full488 (new)')
    else:
        old_w = d.get('wisio', {}).get('metrics', {})
        old_n = d.get('wisio', {}).get('n_samples_evaluated', 0)
        _row('WisIO', 'Rules',
             old_w.get('micro_f1', 0), old_w.get('macro_f1', 0),
             old_w.get('hamming_loss', 0), old_w.get('subset_accuracy', 0),
             old_n, 'final_metrics (OLD)')

    # Threshold and Majority
    for key, name, type_ in sysmap_post:
        if key in d:
            m = d[key].get('metrics', d[key])
            _row(name, type_,
                 m.get('micro_f1', 0), m.get('macro_f1', 0),
                 m.get('hamming_loss', 0), m.get('subset_accuracy', 0),
                 d[key].get('n_samples_evaluated', 488), 'final_metrics')

    print()
    if ioagent_cost > 0:
        print(f"IOAgent (GPT-4o) total cost: ${ioagent_cost:.2f}")
    print()


def main():
    logger.info("=== Merging 28 IOAgent chunks ===")
    merge_ioagent_chunks()
    compare_with_other_5()


if __name__ == '__main__':
    main()
