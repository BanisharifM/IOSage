#!/bin/bash
#SBATCH --job-name=cl_extended
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/closed_loop/extended_%j.out
#SBATCH --error=results/closed_loop/extended_%j.err

# =============================================================================
# Extended Closed-Loop Validation: mdtest + multi-node IOR pairs
# =============================================================================
# Runs 4 new validation pairs beyond the original 3 IOR-only pairs:
#   1. mdtest_metadata_storm  (metadata_intensity) - shared vs unique dir
#   2. mdtest_fpp_explosion   (file_strategy)       - 5000 vs 50 files/rank
#   3. ior_collective_vs_independent (interface_choice) - 64 ranks, 4 nodes
#   4. ior_small_to_large_direct (access_granularity)   - 4KB vs 1MB O_DIRECT
#
# This script is an orchestrator that submits child SLURM jobs for each
# benchmark pair and waits for completion. Run from project root:
#   sbatch scripts/run_closed_loop_extended.slurm
#
# Or run specific pairs:
#   sbatch scripts/run_closed_loop_extended.slurm mdtest_metadata_storm
#   sbatch scripts/run_closed_loop_extended.slurm ior_collective_vs_independent
# =============================================================================

echo "============================================================"
echo "Extended Closed-Loop Validation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Environment
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

# Configuration
ACCOUNT="bdau-delta-cpu"
PARTITION="cpu"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ioprescriber"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber"
RESULTS_DIR="results/closed_loop"
WALLTIME="00:30:00"

mkdir -p "$DARSHAN_LOGPATH" "$RESULTS_DIR" "$SCRATCH"

# Which pairs to run (all 4 extended, or user-specified subset)
if [ -n "$1" ]; then
    PAIRS="$@"
else
    PAIRS="mdtest_metadata_storm mdtest_fpp_explosion ior_collective_vs_independent ior_small_to_large_direct"
fi

echo "Pairs to validate: $PAIRS"
echo ""

# Helper: submit a child SLURM job and return the job ID
submit_job() {
    local SCRIPT_PATH="$1"
    local JOB_ID
    JOB_ID=$(sbatch "$SCRIPT_PATH" 2>&1 | awk '{print $NF}')
    echo "$JOB_ID"
}

# Helper: wait for a SLURM job to complete
wait_for_job() {
    local JOB_ID="$1"
    local TIMEOUT="${2:-1800}"
    local ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        STATE=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
        case "$STATE" in
            COMPLETED) echo "COMPLETED"; return 0 ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY)
                echo "$STATE"; return 1 ;;
        esac
        sleep 30
        ELAPSED=$((ELAPSED + 30))
    done
    echo "TIMEOUT"; return 1
}

# ===========================================================================
# Generate and submit child scripts for each pair
# ===========================================================================

run_pair() {
    local PAIR_NAME="$1"
    local BAD_CMD="$2"
    local GOOD_CMD="$3"
    local BENCH_TYPE="$4"
    local NODES="${5:-1}"
    local NTASKS="${6:-16}"

    echo "------------------------------------------------------------"
    echo "PAIR: $PAIR_NAME (${BENCH_TYPE}, ${NODES}N x ${NTASKS}T)"
    echo "  Bad:  $BAD_CMD"
    echo "  Good: $GOOD_CMD"
    echo "------------------------------------------------------------"

    # Module load (mdtest is bundled with IOR)
    local MODULE_LOAD="module load ior/3.3.0-gcc13.3.1"

    if [ "$BENCH_TYPE" = "mdtest" ]; then
        local CLEANUP_CMD="rm -rf ${SCRATCH}/mdtest_out* 2>/dev/null || true"
        local SCRATCH_SETUP="mkdir -p ${SCRATCH}/mdtest_out"
    else
        local CLEANUP_CMD="rm -f ${SCRATCH}/ior_test_file* 2>/dev/null || true"
        local SCRATCH_SETUP="mkdir -p ${SCRATCH}"
    fi

    # --- Bad config script ---
    local BAD_SCRIPT="${RESULTS_DIR}/iop_bad_${PAIR_NAME}.slurm"
    cat > "$BAD_SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=bad_${PAIR_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=${WALLTIME}
#SBATCH --output=${RESULTS_DIR}/bad_${PAIR_NAME}_%j.out
#SBATCH --error=${RESULTS_DIR}/bad_${PAIR_NAME}_%j.err

${MODULE_LOAD}
export DARSHAN_LOGPATH="${DARSHAN_LOGPATH}"
mkdir -p "\${DARSHAN_LOGPATH}"
cleanup() { ${CLEANUP_CMD}; }
trap cleanup EXIT

echo "Pair: ${PAIR_NAME} (BAD)"
echo "Command: ${BAD_CMD}"
echo "Date: \$(date)"
echo "Nodes: ${NODES}, Tasks: ${NTASKS}"

${SCRATCH_SETUP}

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${BAD_CMD}

echo "Exit code: \$?"
echo "Completed: \$(date)"
ls -t \${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1
SLURM_EOF
    chmod 755 "$BAD_SCRIPT"

    # --- Good config script ---
    local GOOD_SCRIPT="${RESULTS_DIR}/iop_good_${PAIR_NAME}.slurm"
    cat > "$GOOD_SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=good_${PAIR_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=${WALLTIME}
#SBATCH --output=${RESULTS_DIR}/good_${PAIR_NAME}_%j.out
#SBATCH --error=${RESULTS_DIR}/good_${PAIR_NAME}_%j.err

${MODULE_LOAD}
export DARSHAN_LOGPATH="${DARSHAN_LOGPATH}"
mkdir -p "\${DARSHAN_LOGPATH}"
cleanup() { ${CLEANUP_CMD}; }
trap cleanup EXIT

echo "Pair: ${PAIR_NAME} (GOOD)"
echo "Command: ${GOOD_CMD}"
echo "Date: \$(date)"
echo "Nodes: ${NODES}, Tasks: ${NTASKS}"

${SCRATCH_SETUP}

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${GOOD_CMD}

echo "Exit code: \$?"
echo "Completed: \$(date)"
ls -t \${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1
SLURM_EOF
    chmod 755 "$GOOD_SCRIPT"

    # Submit bad config
    echo "  Submitting bad config..."
    local BAD_JOB
    BAD_JOB=$(submit_job "$BAD_SCRIPT")
    echo "  Bad job ID: $BAD_JOB"

    local BAD_STATE
    BAD_STATE=$(wait_for_job "$BAD_JOB" 1800)
    echo "  Bad job state: $BAD_STATE"

    # Submit good config
    echo "  Submitting good config..."
    local GOOD_JOB
    GOOD_JOB=$(submit_job "$GOOD_SCRIPT")
    echo "  Good job ID: $GOOD_JOB"

    local GOOD_STATE
    GOOD_STATE=$(wait_for_job "$GOOD_JOB" 1800)
    echo "  Good job state: $GOOD_STATE"

    echo "  Done: bad=$BAD_STATE, good=$GOOD_STATE"
    echo ""
}

# ===========================================================================
# Pair definitions
# ===========================================================================

for PAIR in $PAIRS; do
    case "$PAIR" in
        mdtest_metadata_storm)
            run_pair "mdtest_metadata_storm" \
                "mdtest -n 10000 -w 100 -e 100 -F -d ${SCRATCH}/mdtest_out" \
                "mdtest -n 10000 -w 100 -e 100 -F -u -d ${SCRATCH}/mdtest_out" \
                "mdtest" 1 16
            ;;
        mdtest_fpp_explosion)
            run_pair "mdtest_fpp_explosion" \
                "mdtest -n 5000 -w 4096 -e 4096 -F -u -d ${SCRATCH}/mdtest_out" \
                "mdtest -n 50 -w 4096 -e 4096 -F -u -d ${SCRATCH}/mdtest_out" \
                "mdtest" 1 128
            ;;
        ior_collective_vs_independent)
            run_pair "ior_collective_vs_independent" \
                "ior -a MPIIO -t 1048576 -b 100M -s 10 -e -C -w -r -o ${SCRATCH}/ior_test_file" \
                "ior -a MPIIO -t 1048576 -b 100M -s 10 -c -w -r -o ${SCRATCH}/ior_test_file" \
                "ior" 4 64
            ;;
        ior_small_to_large_direct)
            run_pair "ior_small_to_large_direct" \
                "ior -a POSIX -t 4096 -b 1M -s 100 -F -e -C -w -r -O useO_DIRECT=1 -o ${SCRATCH}/ior_test_file" \
                "ior -a POSIX -t 1048576 -b 100M -s 10 -F -e -C -w -r -O useO_DIRECT=1 -o ${SCRATCH}/ior_test_file" \
                "ior" 1 16
            ;;
        *)
            echo "WARNING: Unknown pair '$PAIR', skipping."
            ;;
    esac
done

echo "============================================================"
echo "All extended pairs submitted."
echo "Check results in: ${RESULTS_DIR}/"
echo "Parse Darshan logs from: ${DARSHAN_LOGPATH}/"
echo "Completed: $(date)"
echo "============================================================"
