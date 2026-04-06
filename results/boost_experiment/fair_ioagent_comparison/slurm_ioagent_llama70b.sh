#!/bin/bash
#SBATCH --job-name=ioagent_llama70b
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=16g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/fair_ioagent_comparison/ioagent_llama70b/slurm_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/fair_ioagent_comparison/ioagent_llama70b/slurm_%j.err

echo "=== IOAgent (IONavigator) with Llama-3.1-70B (fair comparison) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH

source /work/hdd/bdau/mbanisharifdehkordi/SC_2026/.env

eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/fair_ioagent_comparison/run_ioagent_multimodel.py \
    --model "meta-llama/llama-3.1-70b-instruct" \
    --n-per-bench 10 \
    --seed 42 \
    --output-dir results/boost_experiment/fair_ioagent_comparison/ioagent_llama70b

echo "End: $(date)"
echo "Exit code: $?"
