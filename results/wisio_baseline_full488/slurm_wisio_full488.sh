#!/bin/bash
#SBATCH --job-name=wisio_488
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/wisio_baseline_full488/slurm_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/wisio_baseline_full488/slurm_%j.err

echo "=== WisIO baseline on full 488-sample test set ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
source /work/hdd/bdau/mbanisharifdehkordi/SC_2026/.env
eval "$(conda shell.bash hook)"
conda activate /projects/bdau/envs/sc2026

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

/projects/bdau/envs/sc2026/bin/python results/wisio_baseline_full488/run_wisio_full488.py

echo "End: $(date)"
echo "Exit code: $?"
