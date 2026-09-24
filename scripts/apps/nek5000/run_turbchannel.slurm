#!/bin/bash
#SBATCH --job-name=nek_tc
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/apps/nek5000/jobs/nek_tc_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/apps/nek5000/jobs/nek_tc_%j.err
#SBATCH --export=NONE
# One Nek5000 turbChannel run of the application study (docs/9_fresh_start/TASK_01).
#
# Usage:
#   sbatch --export=CASE_ID=N2,CASE_ROLE=problem,CASE_KIND=constructed,REPEAT=3,ATTEMPT=<name>,\
#          MANIFEST=<path>,EVIDENCE_DIR=<dir>[,NODARSHAN=1] [--output=...] \
#          run_turbchannel.slurm <numSteps> <writeInterval> <writeNFiles> <writeDoublePrecision yes|no>
#
# The case, binary and mesh live outside the repository (APPS_ROOT/nek5000); this script and
# scripts/apps/nek5000/evidence.py hold the experiment logic. The run is timed around srun
# only. Field files are read for the correctness and I/O checks and then removed with the
# scratch directory by the cleanup trap; the log, the edited .par and the evidence JSON are
# kept under EVIDENCE_DIR/logs_<jobid>/.
export SLURM_EXPORT_ENV=ALL   # the batch environment (--export=NONE) is what the step gets
set -u
for v in CASE_ID CASE_ROLE CASE_KIND REPEAT ATTEMPT MANIFEST EVIDENCE_DIR; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v must be passed with --export"; exit 2; }
done
[ $# -eq 4 ] || { echo "ERROR: four positional arguments required: numSteps writeInterval writeNFiles dp"; exit 2; }
NSTEPS=$1; WINT=$2; NFILES=$3; DP=$4
IOSAGE_ROOT=${IOSAGE_ROOT:-/work/hdd/bdau/mbanisharifdehkordi/IOSage}
export DARSHAN_CONFIG_PATH=${DARSHAN_CONFIG_PATH:-$IOSAGE_ROOT/configs/darshan_runtime.conf}
source /work/hdd/bdau/mbanisharifdehkordi/apps/common/env.sh && iosage_env_check || { echo "ERROR: application paths invalid"; exit 2; }
IOSAGE_ENV=${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}
[ -x "$IOSAGE_ENV/bin/python" ] || { echo "ERROR: python missing in $IOSAGE_ENV"; exit 2; }
APP=$APPS_ROOT/nek5000
source $APP/modules.sh
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null; export SLURM_CPUS_PER_TASK=1
CASE=$APP/runs/turbChannel_baseline
SCRATCH=$SCRATCH_ROOT/nek5000_${SLURM_JOB_ID}
export DARSHAN_LOGPATH=$APP_LOG_ROOT/nek5000
LOGD=$EVIDENCE_DIR/logs_${SLURM_JOB_ID}
mkdir -p "$DARSHAN_LOGPATH" "$SCRATCH" "$LOGD" || exit 2
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT
for f in nek5000 turbChannel.re2 turbChannel.ma2 turbChannel.usr SIZE turbChannel.amg.dat turbChannel.amg_AfP.dat turbChannel.amg_Aff.dat turbChannel.amg_W.dat; do
  [ -e "$CASE/$f" ] || { echo "ERROR: missing case file $CASE/$f"; exit 2; }; ln -sfn "$CASE/$f" "$SCRATCH/$f"
done
cp "$CASE/turbChannel.par" "$SCRATCH/turbChannel.par"; cd "$SCRATCH"
sed -i -e "s/^stopAt = .*/stopAt = numSteps/" -e "s/^endTime = .*/numSteps = $NSTEPS/" \
       -e "s/^writeControl = .*/writeControl = timeStep/" -e "s/^writeInterval = .*/writeInterval = $WINT/" turbChannel.par
grep -q "^writeNFiles" turbChannel.par && { echo "ERROR: base .par already sets writeNFiles"; exit 2; }
sed -i "/^writeInterval/a writeNFiles = $NFILES\nwriteDoublePrecision = $DP" turbChannel.par
printf "turbChannel\n%s/\n" "$SCRATCH" > SESSION.NAME
echo "job $SLURM_JOB_ID host $(hostname) case $CASE_ID role $CASE_ROLE repeat $REPEAT attempt $ATTEMPT numSteps $NSTEPS writeInterval $WINT writeNFiles $NFILES dp $DP date $(date)"
grep -E "^(stopAt|numSteps|writeControl|writeInterval|writeNFiles|writeDoublePrecision)" turbChannel.par
PRELOAD=""; [ -z "${NODARSHAN:-}" ] && PRELOAD=",LD_PRELOAD=$DARSHAN_LIB"
START_ISO=$(date -Is); T0=$(date +%s.%N)
srun --cpu-bind=none --export=ALL$PRELOAD ./nek5000 > logfile 2>&1; RC=$?
T1=$(date +%s.%N); WALL=$(echo "$T1 - $T0" | bc)
echo "exit $RC wall_s $WALL nodarshan ${NODARSHAN:+yes}"
grep -E "Step +$NSTEPS,|total elapsed time|run successful|ERROR" logfile | tail -4
cp logfile turbChannel.par "$LOGD/"
SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
COMMIT=$(git -C "$IOSAGE_ROOT" rev-parse HEAD)
PYTHONNOUSERSITE=1 "$IOSAGE_ENV/bin/python" "$IOSAGE_ROOT/scripts/apps/nek5000/evidence.py" \
  --scratch "$SCRATCH" --case-dir "$CASE" --logfile "$SCRATCH/logfile" \
  --nsteps "$NSTEPS" --write-interval "$WINT" --nfiles "$NFILES" --double-precision "$DP" \
  --darshan-logpath "$DARSHAN_LOGPATH" ${NODARSHAN:+--nodarshan} \
  --manifest "$MANIFEST" --evidence-json "$LOGD/evidence.json" \
  --app nek5000 --attempt "$ATTEMPT" --case "$CASE_ID" --role "$CASE_ROLE" --kind "$CASE_KIND" --repeat "$REPEAT" \
  --jobid "$SLURM_JOB_ID" --nodes "$SLURM_JOB_NUM_NODES" --ntasks "$SLURM_NTASKS" --nodelist "$SLURM_JOB_NODELIST" \
  --start-iso "$START_ISO" --wall-s "$WALL" --rc "$RC" --script "$SCRIPT" --binary "$CASE/nek5000" \
  --darshan-config "$DARSHAN_CONFIG_PATH" --iosage-commit "$COMMIT" --scratch-root "$SCRATCH_ROOT"
ERC=$?
ls -t "$DARSHAN_LOGPATH"/*id${SLURM_JOB_ID}-* 2>/dev/null | sed 's/^/darshan: /'
[ $RC -eq 0 ] || exit $RC
exit $ERC
