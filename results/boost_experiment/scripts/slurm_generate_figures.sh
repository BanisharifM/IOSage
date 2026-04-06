#!/bin/bash
#SBATCH --job-name=gen_figures
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/gen_figures_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/gen_figures_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

# 1. Regenerate t-SNE figure (with 689 benchmark samples)
/projects/bdau/envs/sc2026/bin/python -c "
import sys
sys.path.insert(0, '.')
from scripts.generate_benchmark_figures import fig_domain_shift_tsne
fig_domain_shift_tsne()
print('t-SNE figure generated successfully')
"

# 2. Regenerate SHAP figure from boost experiment SHAP values
/projects/bdau/envs/sc2026/bin/python -c "
import sys, pickle, numpy as np
sys.path.insert(0, '.')
from pathlib import Path

PROJECT = Path('/work/hdd/bdau/mbanisharifdehkordi/SC_2026')
shap_path = PROJECT / 'results/boost_experiment/full_evaluation/shap_values.pkl'
model_path = PROJECT / 'results/boost_experiment/new_models/xgboost_biquality_w100_seed42.pkl'

if shap_path.exists():
    with open(shap_path, 'rb') as f:
        shap_data = pickle.load(f)
    print(f'SHAP data type: {type(shap_data)}, keys: {list(shap_data.keys()) if isinstance(shap_data, dict) else \"not dict\"}')

    if isinstance(shap_data, dict) and 'shap_dict' in shap_data:
        raw = shap_data['shap_dict']
        # Handle tuple (shap_dict, X_sample) format
        shap_dict = raw[0] if isinstance(raw, tuple) else raw
        feature_names = shap_data.get('feature_names', [f'f{i}' for i in range(157)])
        from src.models.attribution import plot_global_bar
        output_path = PROJECT / 'paper/figures/fig_shap_global_bar.pdf'
        try:
            plot_global_bar(shap_dict, feature_names, output_path)
            print('SHAP figure generated successfully')
        except Exception as e:
            print(f'SHAP figure error: {e}')
            import traceback; traceback.print_exc()
    else:
        print(f'Unexpected SHAP data structure')
else:
    print(f'SHAP values not found at {shap_path}')
"

echo "Figure generation complete"
