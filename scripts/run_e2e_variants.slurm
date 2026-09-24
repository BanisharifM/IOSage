#!/bin/bash
#SBATCH --job-name=e2e_variants
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_%x_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_%x_%j.err
#SBATCH --export=NONE
export SLURM_EXPORT_ENV=ALL
# Usage: sbatch --job-name=e2e_<variant> scripts/run_e2e_variants.slurm <variant> [repeats]
#   variant: default | pathological | ultra_optimized   (benchmarks/e2e/write_3d_nc4[_variant])
# E2E NetCDF-4 write kernel (Lofstead et al. lineage, as used by Drishti/AIIO/ION): 8x8x1
# processes, 32x32x128 local grid, 10 double variables (~640 MB per run). Each repeat writes a
# fresh file; one Darshan log per repeat (job id + step in the name).
set -euo pipefail
PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/IOSage
E2E_DIR=$PROJECT_DIR/benchmarks/e2e
source $E2E_DIR/modules.sh
source $E2E_DIR/job_lib.sh
e2e_build
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null; export SLURM_CPUS_PER_TASK=1
VARIANT=${1:?variant}; REPEATS=${2:-3}
case $VARIANT in
  default) BIN=$E2E_DIR/build/write_3d_nc4 ;;
  pathological|ultra_optimized) BIN=$E2E_DIR/build/write_3d_nc4_$VARIANT ;;
  *) echo "invalid variant: $VARIANT" >&2; exit 2 ;;
esac
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || { echo "repeats must be a positive integer" >&2; exit 2; }
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 2; }
export DARSHAN_LOGPATH=$PROJECT_DIR/data/benchmark_logs/e2e
export DARSHAN_CONFIG_PATH=$PROJECT_DIR/configs/darshan_runtime.conf
DARSHAN_LIB=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so
SCRATCH=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e/${VARIANT}_$SLURM_JOB_ID
RESULTS_DIR=$PROJECT_DIR/results/resubmission/closed_loop/$SLURM_JOB_ID
MANIFEST=$RESULTS_DIR/e2e_${VARIANT}_${SLURM_JOB_ID}.tsv
mkdir -p "$DARSHAN_LOGPATH" "$SCRATCH" "$RESULTS_DIR"; cd "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
echo "job $SLURM_JOB_ID variant $VARIANT repeats $REPEATS binary $BIN host $(hostname) date $(date)"
for ((r=1; r<=REPEATS; r++)); do
  rm -f "$SCRATCH/out_$r.nc4"
  T0=$(date +%s.%N)
  if [[ $VARIANT == ultra_optimized ]]; then
    FIX_ID=netcdf_disable_fill_process_chunks_no_sync
  else
    FIX_ID=none
  fi
  e2e_run_and_record "$VARIANT" "$FIX_ID" "$BIN" "$SCRATCH/out_$r" "$MANIFEST" \
    8 8 1 32 32 128 > "run_$r.log" 2>&1
  T1=$(date +%s.%N)
  echo "repeat $r start $T0 end $T1 file_bytes $(stat -c %s "$SCRATCH/out_$r.nc4")"
  awk 'tolower($0) ~ /error|time|elapsed/ {print; count++} count == 2 {exit}' "run_$r.log"
done
[[ $(wc -l < "$MANIFEST") -eq $((REPEATS + 1)) ]] || { echo "incomplete manifest: $MANIFEST" >&2; exit 92; }
cat "$MANIFEST"
