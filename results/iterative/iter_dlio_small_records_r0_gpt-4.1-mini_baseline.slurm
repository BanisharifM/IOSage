#!/bin/bash
#SBATCH --job-name=iter_dlio_small_records_r0_gpt-4.1-mini_baseline
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdau-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_dlio_small_records_r0_gpt-4.1-mini_baseline_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_dlio_small_records_r0_gpt-4.1-mini_baseline_%j.err

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
export DARSHAN_MODMEM=16
export DARSHAN_ENABLE_NONMPI=1
# Darshan config: exclude Python library paths, increase record limit to 8192
# Without this, Python's 500+ .so/.pyc imports fill all 1024 POSIX record slots
# before DLIO reads any actual data files, making the Darshan log useless.
export DARSHAN_CONFIG_PATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/configs/darshan_dlio.conf"
mkdir -p "${DARSHAN_LOGPATH}"

# Fix PyTorch CUDA library conflict on Delta.
# The system anaconda3 ships libnvJitLink.so.12.1.105 (CUDA 12.1) which only
# provides symbols up to __nvJitLinkCreate_12_1.  PyTorch 2.8+cu128 installs
# nvidia-cusparse-cu12 12.5 which requires __nvJitLinkCreate_12_8 from the
# pip-installed nvidia-nvjitlink-cu12 12.8.  Because /sw/external/python/
# anaconda3/lib appears early in LD_LIBRARY_PATH, the old library is loaded
# first and the import fails with "undefined symbol __nvJitLinkCreate_12_8".
# Prepending the pip-installed NVIDIA lib paths ensures the correct versions
# are found before the stale system copies.
NVIDIA_LIB_BASE="/projects/bdau/envs/sc2026/lib/python3.9/site-packages/nvidia"
NVIDIA_LIBS=""
for subdir in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$subdir" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$subdir"
done
export LD_LIBRARY_PATH="${NVIDIA_LIBS:+$NVIDIA_LIBS:}${LD_LIBRARY_PATH:-}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=4

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_small_records_r0_gpt-4.1-mini_baseline"
mkdir -p "$JOB_SCRATCH"

# Cleanup DLIO data on exit
cleanup() { rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - DLIO Execution"
echo "Job: iter_dlio_small_records_r0_gpt-4.1-mini_baseline"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Step 1: Generate data (no Darshan needed for data gen)
echo ""
echo "=== DLIO Data Generation Phase ==="
srun --cpu-bind=none --export=ALL \
    /projects/bdau/envs/sc2026/bin/dlio_benchmark workload=unet3d_v100 ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.dataset.data_folder=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_small_records_r0_gpt-4.1-mini_baseline ++workload.dataset.record_length=1024 ++workload.dataset.record_length_stdev=0 ++workload.dataset.num_files_train=500 ++workload.dataset.num_samples_per_file=1 ++workload.reader.batch_size=1 ++workload.reader.read_threads=1 ++workload.train.computation_time=0.01 ++workload.train.epochs=3 ++workload.train.seed=42 ++workload.dataset.format=npz

echo "Data generation complete at $(date)"

# Step 2: Run training (this generates the Darshan log we want)
echo ""
echo "=== DLIO Training Phase (with Darshan) ==="
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    /projects/bdau/envs/sc2026/bin/dlio_benchmark workload=unet3d_v100 ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.dataset.data_folder=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_dlio_small_records_r0_gpt-4.1-mini_baseline ++workload.dataset.record_length=1024 ++workload.dataset.record_length_stdev=0 ++workload.dataset.num_files_train=500 ++workload.dataset.num_samples_per_file=1 ++workload.reader.batch_size=1 ++workload.reader.read_threads=1 ++workload.train.computation_time=0.01 ++workload.train.epochs=3 ++workload.train.seed=42 ++workload.dataset.format=npz

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
