#!/bin/bash
#SBATCH --job-name=detect_logs
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/iterative/detect_%j.out
# Run the ML detector on Darshan logs as a batch job (self-contained /work env).
# Usage: sbatch scripts/run_detector_logs.slurm <model.pkl> <output.json> <log>...
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
export PATH="$IOSAGE_ENV/bin:$PATH" PYTHONNOUSERSITE=1; unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/IOSage || exit 1
MODEL="$1"; OUT="$2"; shift 2
echo "host $(hostname) model $MODEL"
"$IOSAGE_ENV/bin/python" scripts/detect_darshan_logs.py --model "$MODEL" --output "$OUT" "$@"
