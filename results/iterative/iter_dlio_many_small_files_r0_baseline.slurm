#!/bin/bash
#SBATCH --job-name=iter_dlio_many_small_files_r0_baseline
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_dlio_many_small_files_r0_baseline_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_dlio_many_small_files_r0_baseline_%j.err

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
export DARSHAN_MODMEM=4
export DARSHAN_ENABLE_NONMPI=1
mkdir -p "${DARSHAN_LOGPATH}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=4

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_many_small_files_r0_baseline"
mkdir -p "$JOB_SCRATCH"

# Cleanup DLIO data on exit
cleanup() { rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "Track C Iterative Optimization - DLIO Execution"
echo "Job: iter_dlio_many_small_files_r0_baseline"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Step 1: Generate data (no Darshan needed for data gen)
echo ""
echo "=== DLIO Data Generation Phase ==="
srun --export=ALL \
    /projects/bdau/envs/sc2026/bin/dlio_benchmark workload=unet3d_v100 ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.dataset.data_folder=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_many_small_files_r0_baseline ++workload.dataset.record_length=4096 ++workload.dataset.num_files_train=10000 ++workload.dataset.num_samples_per_file=1 ++workload.reader.batch_size=1 ++workload.reader.read_threads=1 ++workload.train.computation_time=0.01 ++workload.train.epochs=2 ++workload.train.seed=42 ++workload.dataset.format=npz

echo "Data generation complete at $(date)"

# Step 2: Run training (this generates the Darshan log we want)
echo ""
echo "=== DLIO Training Phase (with Darshan) ==="
srun --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    /projects/bdau/envs/sc2026/bin/dlio_benchmark workload=unet3d_v100 ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.dataset.data_folder=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_many_small_files_r0_baseline ++workload.dataset.record_length=4096 ++workload.dataset.num_files_train=10000 ++workload.dataset.num_samples_per_file=1 ++workload.reader.batch_size=1 ++workload.reader.read_threads=1 ++workload.train.computation_time=0.01 ++workload.train.epochs=2 ++workload.train.seed=42 ++workload.dataset.format=npz

EXIT_CODE=$?
echo ""
echo "DLIO training complete at $(date), exit code: $EXIT_CODE"

# Report Darshan log location
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*id${SLURM_JOB_ID}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
