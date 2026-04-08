#!/bin/bash
#SBATCH --job-name=ios_claudesonnet
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=8g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/iosage_full488/output_claude-sonnet-4/slurm_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/iosage_full488/output_claude-sonnet-4/slurm_%j.err

echo "=== IOSage full pipeline on 488 traces, model=claude-sonnet-4 ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
source /work/hdd/bdau/mbanisharifdehkordi/SC_2026/.env
eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
mkdir -p results/boost_experiment/iosage_full488/output_claude-sonnet-4

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/iosage_full488/run_iosage_full488.py \
    --model claude-sonnet-4 \
    --output-dir results/boost_experiment/iosage_full488/output_claude-sonnet-4

echo "End: $(date)"
echo "Exit code: $?"
