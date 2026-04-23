#!/bin/bash
#SBATCH --job-name=boost_weak_dims
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/boost_weak_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/boost_weak_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
module load ior/3.3.0-gcc13.3.1

DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ior"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"

mkdir -p "$LOG_DIR" "$SCRATCH/bottleneck" "$SCRATCH/healthy"
export DARSHAN_LOGPATH="$LOG_DIR"

echo "=== Boost Weak Dimensions ==="
echo "Start: $(date)"

# ================================================================
# ACCESS_PATTERN: 24 more runs (various sizes, ranks, shared file)
# Target: 55+37+24 = need ~30 more after what we have
# ================================================================
echo "--- ACCESS PATTERN: random shared file ---"
for size in 4096 16384 65536 262144; do
    for ranks in 4 16 64; do
        for rep in 1 2; do
            echo "Running: access_pattern t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -z -t ${size} -b 10M -s 10 -e -w -r \
                -o ${SCRATCH}/bottleneck/ap_shared_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/bottleneck/ap_shared_t${size}_n${ranks}_r${rep}*
        done
    done
done

# ================================================================
# FILE_STRATEGY: 30 more runs (file-per-process with many ranks)
# Target: 48+30 = 78 total
# ================================================================
echo "--- FILE STRATEGY: file explosion ---"
lfs setstripe -c 1 -S 1M ${SCRATCH}/bottleneck 2>/dev/null || true
for ranks in 32 64 128; do
    for size in 65536 1048576; do
        for rep in 1 2 3; do
            echo "Running: file_strategy n=${ranks} t=${size} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -t ${size} -b 10M -s 10 -F -e -w -r \
                -o ${SCRATCH}/bottleneck/fs_fpp_n${ranks}_t${size}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/bottleneck/fs_fpp_n${ranks}_t${size}_r${rep}*
        done
    done
done

# ================================================================
# THROUGHPUT_UTILIZATION: 18 more runs (fsync + single OST)
# Target: 63+18 = 81 total
# ================================================================
echo "--- THROUGHPUT: fsync heavy + single OST ---"
lfs setstripe -c 1 -S 1M ${SCRATCH}/bottleneck 2>/dev/null || true
for size in 65536 1048576; do
    for ranks in 4 16 32; do
        for rep in 1 2 3; do
            echo "Running: throughput fsync t=${size} n=${ranks} rep=${rep}"
            srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
                ior -a POSIX -t ${size} -b 100M -s 4 -e -w -r -Y \
                -o ${SCRATCH}/bottleneck/tp_fsync_t${size}_n${ranks}_r${rep} 2>&1 || true
            rm -f ${SCRATCH}/bottleneck/tp_fsync_t${size}_n${ranks}_r${rep}*
        done
    done
done

# ================================================================
# HEALTHY: 12 more (large collective MPI-IO at scale, reference)
# ================================================================
echo "--- HEALTHY: large collective at scale ---"
unset MPICH_MPIIO_HINTS
lfs setstripe -c -1 -S 1M ${SCRATCH}/healthy 2>/dev/null || true
for ranks in 32 64 128; do
    for rep in 1 2; do
        echo "Running: healthy collective n=${ranks} rep=${rep}"
        srun -n ${ranks} --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
            ior -a MPIIO -c -t 4194304 -b 1G -s 4 -e -w -r \
            -o ${SCRATCH}/healthy/healthy_coll_n${ranks}_r${rep} 2>&1 || true
        rm -f ${SCRATCH}/healthy/healthy_coll_n${ranks}_r${rep}*
    done
done

echo "End: $(date)"
echo "Total configs: 24 + 18 + 18 + 6 = 66"
echo "=== Done ==="
