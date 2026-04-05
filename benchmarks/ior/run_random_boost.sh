#!/bin/bash
#SBATCH --job-name=ior_random_boost
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=32g
#SBATCH --output=data/benchmark_results/ior/random_boost_%j.out
#SBATCH --error=data/benchmark_results/ior/random_boost_%j.err

module load ior/3.3.0-gcc13.3.1

DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
LOG_DIR="$PWD/data/benchmark_logs/ior"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/bottleneck"

mkdir -p "$LOG_DIR" "$SCRATCH"
export DARSHAN_LOGPATH="$LOG_DIR"
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable"

echo "=== IOR Random Access Boost (access_pattern=1) ==="
echo "Start: $(date)"

# Random POSIX with various transfer sizes and rank counts
for size in 4096 16384 65536 262144 1048576; do
    for ranks in 4 16 32; do
        for rep in 1 2 3; do
            echo "Running: random_posix t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 100M -s 10 -e -C \
                -o ${SCRATCH}/random_posix_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/random_posix_t${size}_n${ranks}_r${rep}*
        done
    done
done

# Random small (access_pattern + access_granularity)
for size in 512 1024 2048; do
    for ranks in 4 16; do
        for rep in 1 2 3; do
            echo "Running: random_small t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 10M -s 50 -e -C \
                -o ${SCRATCH}/random_small_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/random_small_t${size}_n${ranks}_r${rep}*
        done
    done
done

echo "End: $(date)"
echo "=== Done ==="
