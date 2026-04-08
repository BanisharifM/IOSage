#!/bin/bash
#SBATCH --job-name=rsm_ior_c7
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=16g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/ioagent_gpt4o_full488/output_ior_c7/slurm_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/ioagent_gpt4o_full488/output_ior_c7/slurm_%j.err

echo "=== IOAgent RESUME (unique-file fix): ior_c7 ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
source /work/hdd/bdau/mbanisharifdehkordi/SC_2026/.env
eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/ioagent_gpt4o_full488/run_ioagent_resume_unique.py \
    --model "openai/gpt-4o" \
    --benchmark ior \
    --chunk-idx 7 \
    --n-chunks 8 \
    --output-dir results/boost_experiment/ioagent_gpt4o_full488/output_ior_c7

echo "End: $(date)"
echo "Exit code: $?"
