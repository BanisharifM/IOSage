#!/bin/bash
#SBATCH --job-name=verify_gt
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/verification/verify_gt_%j.out
# Verify every benchmark Darshan sample against its constructed label and write one row per
# sample. Usage: sbatch scripts/verify_ground_truth.slurm [bench-type, default all]
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
export PATH="$IOSAGE_ENV/bin:$PATH" PYTHONNOUSERSITE=1; unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/IOSage || exit 1
BENCH="${1:-all}"
OUT=results/resubmission/verification/ground_truth_${BENCH}_$(date +%F)_${SLURM_JOB_ID}.csv
echo "host $(hostname) commit $(git rev-parse --short HEAD) bench $BENCH report $OUT"
"$IOSAGE_ENV/bin/python" scripts/verify_all_ground_truth.py --bench-type "$BENCH" --report "$OUT"
echo "verify rc=$?"
