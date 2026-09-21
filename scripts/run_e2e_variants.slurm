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
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/e2e/e2e_%x_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/e2e/e2e_%x_%j.err
#SBATCH --export=NONE
export SLURM_EXPORT_ENV=ALL
# Usage: sbatch --job-name=e2e_<variant> scripts/run_e2e_variants.slurm <variant> [repeats]
#   variant: default | pathological | ultra_optimized   (benchmarks/e2e/write_3d_nc4[_variant])
# E2E NetCDF-4 write kernel (Lofstead et al. lineage, as used by Drishti/AIIO/ION): 8x8x1
# processes, 32x32x128 local grid, 10 double variables (~640 MB per run). Each repeat writes a
# fresh file; one Darshan log per repeat (job id + step in the name).
set -u
PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/IOSage
E2E_DIR=$PROJECT_DIR/benchmarks/e2e
source $E2E_DIR/modules.sh
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null; export SLURM_CPUS_PER_TASK=1
VARIANT=${1:?variant}; REPEATS=${2:-3}
case $VARIANT in default) BIN=$E2E_DIR/write_3d_nc4;; *) BIN=$E2E_DIR/write_3d_nc4_$VARIANT;; esac
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 2; }
export DARSHAN_LOGPATH=$PROJECT_DIR/data/benchmark_logs/e2e
export DARSHAN_CONFIG_PATH=$PROJECT_DIR/configs/darshan_runtime.conf
DARSHAN_LIB=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so
SCRATCH=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e/${VARIANT}_$SLURM_JOB_ID
mkdir -p "$DARSHAN_LOGPATH" "$SCRATCH"; cd "$SCRATCH"
echo "job $SLURM_JOB_ID variant $VARIANT repeats $REPEATS binary $BIN host $(hostname) date $(date)"
for r in $(seq 1 $REPEATS); do
  rm -f "$SCRATCH/out_$r.nc4"
  T0=$(date +%s.%N)
  srun --cpu-bind=none --export=ALL,LD_PRELOAD=$DARSHAN_LIB "$BIN" "$SCRATCH/out_$r" 8 8 1 32 32 128 > run_$r.log 2>&1
  RC=$?; T1=$(date +%s.%N)
  echo "repeat $r exit $RC wall_s $(echo "$T1 - $T0" | bc) file_bytes $(stat -c %s "$SCRATCH/out_$r.nc4" 2>/dev/null || echo 0)"
  grep -iE "error|time|elapsed" run_$r.log | head -2
done
ls -t $DARSHAN_LOGPATH/*id${SLURM_JOB_ID}* 2>/dev/null | sed 's/^/darshan: /'
rm -rf "$SCRATCH"
