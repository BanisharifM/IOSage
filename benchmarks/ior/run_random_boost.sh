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
lfs setstripe -c 1 -S 1M "$SCRATCH" 2>/dev/null || true

echo "=== IOR Random Access Boost (access_pattern=1) ==="
echo "Start: $(date)"

# Random POSIX file-per-process (FPP) - -z works with -F
for size in 4096 16384 65536 262144 1048576; do
    for ranks in 4 16; do
        for rep in 1 2 3; do
            echo "Running: random_fpp t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 100M -s 10 -F -e -w -r \
                -o ${SCRATCH}/random_fpp_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/random_fpp_t${size}_n${ranks}_r${rep}*
        done
    done
done

# Random POSIX shared file (no -C flag, just -z)
for size in 4096 65536 1048576; do
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

# Random small FPP (access_pattern + access_granularity)
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
echo "=== Done: $(find $LOG_DIR -name '*.darshan' -newer $0 | wc -l) new logs ==="
