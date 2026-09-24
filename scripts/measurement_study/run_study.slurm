#!/bin/bash
#SBATCH --job-name=mstudy
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/measurement_study/mstudy_%j.out
# Driver for the measurement study; runs as a batch job with all state under /work so that an
# expiring interactive session cannot take the experiment down with it.
# Usage: sbatch scripts/measurement_study/run_study.slurm <script.py> [args...]
export SLURM_EXPORT_ENV=ALL
set -euo pipefail
source /etc/profile
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
export PATH="$IOSAGE_ENV/bin:$PATH" PYTHONNOUSERSITE=1; unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/IOSage
(( $# >= 1 )) || { echo "usage: $0 SCRIPT [ARG ...]" >&2; exit 2; }
echo "host $(hostname) start $(date)"; echo "cmd: $*"
"$IOSAGE_ENV/bin/python" "$@"
echo "end $(date) rc=0"
