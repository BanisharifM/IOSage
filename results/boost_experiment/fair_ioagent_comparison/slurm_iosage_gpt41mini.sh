#!/bin/bash
#SBATCH --job-name=iosage_gpt41mini
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/fair_ioagent_comparison/iosage_gpt41mini/slurm_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/fair_ioagent_comparison/iosage_gpt41mini/slurm_%j.err

echo "=== IOSage with gpt-4.1-mini (fair comparison) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

unset PYTHONPATH

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH

# Load conda
source /work/hdd/bdau/mbanisharifdehkordi/SC_2026/.env

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/fair_ioagent_comparison/run_iosage_gpt41mini.py \
    --model gpt-4.1-mini \
    --n-workloads 12 \
    --n-runs 1

echo "End: $(date)"
echo "Exit code: $?"
