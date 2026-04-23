#!/bin/bash
#SBATCH --job-name=ior_large_scale
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/ior/large_scale_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_results/ior/large_scale_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
module load ior/3.3.0-gcc13.3.1

DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ior"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"

mkdir -p "$LOG_DIR" "$SCRATCH/bottleneck" "$SCRATCH/healthy"
export DARSHAN_LOGPATH="$LOG_DIR"

echo "=== Large-Scale IOR Benchmarks (2 nodes, 128 procs) ==="
echo "Start: $(date)"

# === ACCESS_PATTERN: Random shared file at scale ===
echo "--- Random access, shared file, 128 procs ---"
for size in 4096 65536; do
    for rep in 1 2 3; do
        echo "Running: random_shared_128 t=${size} rep=${rep}"
        srun -n 128 --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
            ior -a POSIX -z -t ${size} -b 10M -s 10 -e -w -r \
            -o ${SCRATCH}/bottleneck/random_128_t${size}_r${rep} 2>&1 || true
        rm -f ${SCRATCH}/bottleneck/random_128_t${size}_r${rep}*
    done
done

# === INTERFACE_CHOICE: POSIX shared vs MPI-IO collective at real scale ===
echo "--- Interface misuse: POSIX shared, 128 procs (should use collective) ---"
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable"
for size in 65536 1048576; do
    for rep in 1 2 3; do
        echo "Running: interface_posix_128 t=${size} rep=${rep}"
        srun -n 128 --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
            ior -a POSIX -t ${size} -b 100M -s 4 -e -w -r \
            -o ${SCRATCH}/bottleneck/interface_posix_128_t${size}_r${rep} 2>&1 || true
        rm -f ${SCRATCH}/bottleneck/interface_posix_128_t${size}_r${rep}*
    done
done

# === HEALTHY: Collective MPI-IO at scale (reference for interface_choice) ===
echo "--- Healthy: MPI-IO collective, 128 procs, large transfers ---"
unset MPICH_MPIIO_HINTS
for size in 1048576 4194304; do
    for rep in 1 2 3; do
        echo "Running: healthy_collective_128 t=${size} rep=${rep}"
        lfs setstripe -c -1 -S 1M ${SCRATCH}/healthy 2>/dev/null || true
        srun -n 128 --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
            ior -a MPIIO -c -t ${size} -b 1G -s 4 -e -w -r \
            -o ${SCRATCH}/healthy/healthy_128_t${size}_r${rep} 2>&1 || true
        rm -f ${SCRATCH}/healthy/healthy_128_t${size}_r${rep}*
    done
done

# === THROUGHPUT: Large data volume (10GB+ total) ===
echo "--- Throughput: single OST, large volume, 64 procs ---"
lfs setstripe -c 1 -S 1M ${SCRATCH}/bottleneck 2>/dev/null || true
for rep in 1 2 3; do
    echo "Running: throughput_large_64 rep=${rep}"
    srun -n 64 --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
        ior -a POSIX -t 1048576 -b 1G -s 4 -F -e -w -r \
        -o ${SCRATCH}/bottleneck/throughput_large_64_r${rep} 2>&1 || true
    rm -f ${SCRATCH}/bottleneck/throughput_large_64_r${rep}*
done

echo "End: $(date)"
echo "=== Done ==="
