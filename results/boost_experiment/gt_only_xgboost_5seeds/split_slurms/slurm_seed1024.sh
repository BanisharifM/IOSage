#!/bin/bash
#SBATCH --job-name=gt_only_seed1024
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=16g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/gt_only_xgboost_5seeds/logs/seed1024_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/gt_only_xgboost_5seeds/logs/seed1024_%j.err

echo "=== GT-only XGBoost training, seed=1024 ==="
echo "Start: $(date)"
echo "Node:  $(hostname)"

# Use the conda env directly without .local_pkgs (avoids numpy version mismatch)
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

/projects/bdau/envs/sc2026/bin/python \
    results/boost_experiment/gt_only_xgboost_5seeds/train_gt_only_single_seed.py \
    --seed 1024 \
    --output-dir results/boost_experiment/gt_only_xgboost_5seeds/output

echo "End: $(date)"
echo "Exit: $?"
