#!/bin/bash
#SBATCH --job-name=wrf_c12
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/apps/wrf/jobs/wrf_c12_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/apps/wrf/jobs/wrf_c12_%j.err
#SBATCH --export=NONE
# One WRF CONUS 12 km run of the application study (docs/9_fresh_start/TASK_02).
#
# Usage:
#   sbatch --export=CASE_ID=W1,CASE_ROLE=problem,CASE_KIND=natural,REPEAT=3,ATTEMPT=<name>,\
#          MANIFEST=<path>,EVIDENCE_DIR=<dir>[,NODARSHAN=1][,KEEP_OUTPUT_DIR=<dir>] run_conus12km.slurm <io_form_history>
#
# io_form_history is the only namelist value the case changes (2 serial NetCDF, 11 PnetCDF).
# The base input (symlinked tables, restart, boundary file and namelist.input.base) lives in
# APPS_ROOT/wrf/runs/conus12km_task02 outside the repository; the experiment logic is this
# script and scripts/apps/wrf/evidence.py. The run is timed around srun only. The history
# file is read for the correctness and I/O checks and removed with the scratch directory by
# the cleanup trap (KEEP_OUTPUT_DIR copies it first, for development runs only); rsl.out.0000,
# rsl.error.0000, the namelist and evidence.json are kept under EVIDENCE_DIR/logs_<jobid>/.
export SLURM_EXPORT_ENV=ALL
set -u
for v in CASE_ID CASE_ROLE CASE_KIND REPEAT ATTEMPT MANIFEST EVIDENCE_DIR; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v must be passed with --export"; exit 2; }
done
[ $# -eq 1 ] || { echo "ERROR: one positional argument required: io_form_history (2 or 11)"; exit 2; }
IO_FORM=$1
case "$IO_FORM" in 2|11) ;; *) echo "ERROR: io_form_history must be 2 or 11, got $IO_FORM"; exit 2;; esac
IOSAGE_ROOT=${IOSAGE_ROOT:-/work/hdd/bdau/mbanisharifdehkordi/IOSage}
export DARSHAN_CONFIG_PATH=${DARSHAN_CONFIG_PATH:-$IOSAGE_ROOT/configs/darshan_runtime.conf}
source /work/hdd/bdau/mbanisharifdehkordi/apps/common/env.sh && iosage_env_check || { echo "ERROR: application paths invalid"; exit 2; }
IOSAGE_ENV=${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}
[ -x "$IOSAGE_ENV/bin/python" ] || { echo "ERROR: python missing in $IOSAGE_ENV"; exit 2; }
APP=$APPS_ROOT/wrf
source $APP/modules.sh
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null; export SLURM_CPUS_PER_TASK=1
BASE=$APP/runs/conus12km_task02
BINARY=$APP/install/v4.8.0-cmake/bin/wrf
SCRATCH=$SCRATCH_ROOT/wrf_${SLURM_JOB_ID}
export DARSHAN_LOGPATH=$APP_LOG_ROOT/wrf
LOGD=$EVIDENCE_DIR/logs_${SLURM_JOB_ID}
for f in "$BASE/namelist.input.base" "$BINARY"; do [ -e "$f" ] || { echo "ERROR: missing $f"; exit 2; }; done
mkdir -p "$DARSHAN_LOGPATH" "$SCRATCH" "$LOGD" || exit 2
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT
# per-job working copy: symlinks to the base input; the namelist is written from the base
cp -rs "$BASE"/. "$SCRATCH"/ 2> "$LOGD/cp.err" || { echo "ABORT: scratch copy failed"; head -3 "$LOGD/cp.err"; exit 3; }
rm -f "$SCRATCH/namelist.input.base" "$SCRATCH/namelist.input"
cd "$SCRATCH"
[ "$(grep -c '^ io_form_history ' "$BASE/namelist.input.base")" -eq 1 ] || { echo "ERROR: base namelist must hold exactly one io_form_history line"; exit 2; }
sed "s/^ io_form_history .*/ io_form_history                     = $IO_FORM,/" "$BASE/namelist.input.base" > namelist.input
for f in wrfrst_d01_2019-11-26_23_00_00 wrfbdy_d01 namelist.input; do
  [ -e "$f" ] || { echo "ERROR: input missing: $f"; exit 2; }; echo "input ok: $f -> $(readlink -f "$f") $(stat -L -c %s "$f")"
done
echo "job $SLURM_JOB_ID host $(hostname) nodes $SLURM_JOB_NODELIST case $CASE_ID role $CASE_ROLE repeat $REPEAT attempt $ATTEMPT io_form_history $IO_FORM date $(date)"
grep -E "^ (io_form_history|io_form_restart|io_form_boundary|nocolons|use_netcdf_classic|history_interval_m|frames_per_outfile|run_hours|time_step|nio_tasks_per_group|nio_groups) " namelist.input
PRELOAD=""; [ -z "${NODARSHAN:-}" ] && PRELOAD=",LD_PRELOAD=$DARSHAN_LIB"
START_ISO=$(date -Is); T0=$(date +%s.%N)
srun --cpu-bind=none --export=ALL$PRELOAD "$BINARY"; RC=$?
T1=$(date +%s.%N); WALL=$(echo "$T1 - $T0" | bc)
echo "exit $RC wall_s $WALL nodarshan ${NODARSHAN:+yes}"
grep -E "Timing for Writing|SUCCESS COMPLETE|FATAL|ERROR" rsl.out.0000 | tail -4; tail -2 rsl.error.0000
cp rsl.out.0000 rsl.error.0000 namelist.input "$LOGD/"
if [ -n "${KEEP_OUTPUT_DIR:-}" ]; then mkdir -p "$KEEP_OUTPUT_DIR"; cp wrfout_d01_* "$KEEP_OUTPUT_DIR"/ 2>/dev/null; cp qr_acr_qg_V4.dat "$KEEP_OUTPUT_DIR"/ 2>/dev/null; fi
SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
COMMIT=$(git -C "$IOSAGE_ROOT" rev-parse HEAD)
PYTHONNOUSERSITE=1 "$IOSAGE_ENV/bin/python" "$IOSAGE_ROOT/scripts/apps/wrf/evidence.py" \
  --scratch "$SCRATCH" --base-dir "$BASE" --io-form-history "$IO_FORM" \
  --darshan-logpath "$DARSHAN_LOGPATH" ${NODARSHAN:+--nodarshan} \
  --manifest "$MANIFEST" --evidence-json "$LOGD/evidence.json" \
  --app wrf --attempt "$ATTEMPT" --case "$CASE_ID" --role "$CASE_ROLE" --kind "$CASE_KIND" --repeat "$REPEAT" \
  --jobid "$SLURM_JOB_ID" --nodes "$SLURM_JOB_NUM_NODES" --ntasks "$SLURM_NTASKS" --nodelist "$SLURM_JOB_NODELIST" \
  --start-iso "$START_ISO" --wall-s "$WALL" --rc "$RC" --script "$SCRIPT" --binary "$BINARY" \
  --wrf-source "$APP/build/WRF-v4.8.0" --darshan-config "$DARSHAN_CONFIG_PATH" --iosage-commit "$COMMIT" --scratch-root "$SCRATCH_ROOT"
ERC=$?
ls -t "$DARSHAN_LOGPATH"/*id${SLURM_JOB_ID}-* 2>/dev/null | sed 's/^/darshan: /'
[ $RC -eq 0 ] || exit $RC
exit $ERC
