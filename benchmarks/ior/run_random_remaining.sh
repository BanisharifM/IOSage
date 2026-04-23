#!/bin/bash
#SBATCH --job-name=ior_random_rem
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/ior/random_remaining_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/ior/random_remaining_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
module load ior/3.3.0-gcc13.3.1

DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ior"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/bottleneck"

mkdir -p "$LOG_DIR" "$SCRATCH"
export DARSHAN_LOGPATH="$LOG_DIR"
lfs setstripe -c 1 -S 1M "$SCRATCH" 2>/dev/null || true

echo "=== IOR Random Remaining Configs (access_pattern=1) ==="
echo "Start: $(date)"

# REMAINING: random_shared configs (skip t=4096 which already ran)
for size in 65536 1048576; do
    for ranks in 4 16; do
        for rep in 1 2 3; do
            echo "Running: random_shared t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 100M -s 10 -e -w -r \
                -o ${SCRATCH}/random_shared_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/random_shared_t${size}_n${ranks}_r${rep}*
        done
    done
done

# REMAINING: random_small FPP configs (none ran yet)
for size in 512 1024 2048; do
    for ranks in 4 16; do
        for rep in 1 2 3; do
            echo "Running: random_small t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 10M -s 50 -F -e -w -r \
                -o ${SCRATCH}/random_small_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/random_small_t${size}_n${ranks}_r${rep}*
        done
    done
done

echo "End: $(date)"
echo "=== Done ==="
