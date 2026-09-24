#!/bin/bash
#SBATCH --job-name=bench_smoke_test
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/data/benchmark_results/smoke_test_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/data/benchmark_results/smoke_test_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/IOSage
BENCH_SCRATCH=/work/hdd/bdau/mbanisharifdehkordi/bench_scratch
DARSHAN_LIB=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so
IOSAGE_ENV=${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}
PYTHON_BIN=$IOSAGE_ENV/bin/python
SMOKE_DIR=$BENCH_SCRATCH/smoke_test_$SLURM_JOB_ID
SMOKE_LOG_DIR=$PROJECT_DIR/data/benchmark_logs/smoke_test
RESULTS_DIR=$PROJECT_DIR/data/benchmark_results
RUN_MANIFEST=$RESULTS_DIR/smoke_test_${SLURM_JOB_ID}.manifest.tsv

source /etc/profile
module load ior/3.3.0-gcc13.3.1
source "$PROJECT_DIR/benchmarks/job_guard.sh"
benchmark_record_executable ior "$RUN_MANIFEST"
benchmark_record_executable mdtest "$RUN_MANIFEST"
mkdir -p "$SMOKE_DIR" "$SMOKE_LOG_DIR" "$RESULTS_DIR"
cleanup() { rm -rf "$SMOKE_DIR"; }
trap cleanup EXIT
export DARSHAN_LOGPATH=$SMOKE_LOG_DIR
export DARSHAN_CONFIG_PATH=$PROJECT_DIR/configs/darshan_runtime.conf

SMALL_DIR=$SMOKE_DIR/single_ost
HEALTHY_DIR=$SMOKE_DIR/full_stripe
mkdir -p "$SMALL_DIR" "$HEALTHY_DIR"
lfs setstripe -c 1 -S 1M "$SMALL_DIR"
lfs setstripe -c -1 -S 1M "$HEALTHY_DIR"
[[ $(lfs getstripe -c "$SMALL_DIR") == 1 ]] || { echo "single-OST setup failed" >&2; exit 3; }
[[ $(lfs getstripe -c "$HEALTHY_DIR") == -1 ]] || { echo "full-stripe setup failed" >&2; exit 3; }

PASS=0
FAIL=0
TESTS=0

run_ior_test()
{
    local test_name=$1
    local labels=$2
    local output_dir=$3
    local buffering=$4
    local tasks=$5
    shift 5
    [[ $1 == -- ]] || { echo "internal smoke-test argument error" >&2; exit 2; }
    shift
    local output_prefix=$output_dir/${test_name}_output
    local run_output=$SMOKE_DIR/${test_name}.out
    local report=$RESULTS_DIR/smoke_${test_name}_${SLURM_JOB_ID}.json
    local log_path

    ((TESTS += 1))
    case $buffering in
        enabled)
            export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"
            ;;
        disabled)
            export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable:romio_ds_write=disable:romio_ds_read=disable"
            ;;
        none)
            unset MPICH_MPIIO_HINTS
            ;;
        *)
            echo "invalid buffering mode: $buffering" >&2
            exit 2
            ;;
    esac

    if ! benchmark_run "$test_name" ior "$RUN_MANIFEST"         srun --ntasks="$tasks" --export="ALL,LD_PRELOAD=$DARSHAN_LIB" "$@" -o "$output_prefix"         >"$run_output" 2>&1; then
        echo "$test_name: IOR run failed"
        tail -20 "$run_output"
        ((FAIL += 1))
        return 0
    fi
    log_path=$(tail -n 1 "$RUN_MANIFEST" | cut -f4)
    if "$PYTHON_BIN" "$PROJECT_DIR/scripts/verify_smoke_scenario.py"         --log "$log_path" --labels "$labels" --output "$report"; then
        echo "$test_name: PASS ($labels)"
        ((PASS += 1))
    else
        echo "$test_name: label verification failed; report $report"
        ((FAIL += 1))
    fi
    rm -f "${output_prefix}"*
}

run_ior_test small_io access_granularity=1 "$HEALTHY_DIR" none 4 --     ior -a POSIX -t 512 -b 64K -s 400 -F -e -C -w -r
run_ior_test random_io access_pattern=1 "$HEALTHY_DIR" none 4 --     ior -a POSIX -t 4096 -b 10M -s 10 -z -F -e -C -w -r --posix.odirect
run_ior_test interface_misuse interface_choice=1 "$HEALTHY_DIR" disabled 4 --     ior -a MPIIO -t 65536 -b 100M -s 4 -e -C -w -r
run_ior_test misaligned request_alignment=1 "$HEALTHY_DIR" none 4 --     ior -a POSIX -t 1000 -b 1000000 -s 50 -F -e -C -w -r
run_ior_test healthy_collective healthy=1 "$HEALTHY_DIR" enabled 1 --     ior -a MPIIO -t 4194304 -b 100M -s 4 -c -e -w -r

((TESTS += 1))
MDTEST_DIR=$SMOKE_DIR/mdtest
mkdir -p "$MDTEST_DIR"
MDTEST_REPORT=$RESULTS_DIR/smoke_mdtest_${SLURM_JOB_ID}.json
if benchmark_run mdtest_metadata mdtest "$RUN_MANIFEST"     srun --export="ALL,LD_PRELOAD=$DARSHAN_LIB"     mdtest -n 100 -w 0 -e 0 -F -d "$MDTEST_DIR"; then
    LOG_PATH=$(tail -n 1 "$RUN_MANIFEST" | cut -f4)
    if "$PYTHON_BIN" "$PROJECT_DIR/scripts/verify_smoke_scenario.py"         --log "$LOG_PATH" --labels metadata_intensity=1 --output "$MDTEST_REPORT"; then
        echo "mdtest_metadata: PASS"
        ((PASS += 1))
    else
        echo "mdtest_metadata: label verification failed; report $MDTEST_REPORT"
        ((FAIL += 1))
    fi
else
    echo "mdtest_metadata: run failed"
    ((FAIL += 1))
fi

((TESTS += 1))
MDTEST_FILES_DIR=$SMOKE_DIR/mdtest_many_small
mkdir -p "$MDTEST_FILES_DIR"
MDTEST_FILES_REPORT=$RESULTS_DIR/smoke_mdtest_many_small_${SLURM_JOB_ID}.json
if benchmark_run mdtest_many_small mdtest "$RUN_MANIFEST"     srun --export="ALL,LD_PRELOAD=$DARSHAN_LIB"     mdtest -n 300 -w 4096 -e 4096 -F -u -d "$MDTEST_FILES_DIR"; then
    LOG_PATH=$(tail -n 1 "$RUN_MANIFEST" | cut -f4)
    if "$PYTHON_BIN" "$PROJECT_DIR/scripts/verify_smoke_scenario.py"         --log "$LOG_PATH" --labels file_strategy=1 --output "$MDTEST_FILES_REPORT"; then
        echo "mdtest_many_small: PASS"
        ((PASS += 1))
    else
        echo "mdtest_many_small: label verification failed; report $MDTEST_FILES_REPORT"
        ((FAIL += 1))
    fi
else
    echo "mdtest_many_small: run failed"
    ((FAIL += 1))
fi

echo "Smoke results: $PASS passed, $FAIL failed, $TESTS total"
[[ $TESTS -eq 7 && $PASS -eq 7 && $FAIL -eq 0 ]] || exit 1
echo "ALL TESTS PASSED. Ready for full sweep."
