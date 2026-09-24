#!/bin/bash
#SBATCH --job-name=iter_ablation
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/ablation_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/ablation_%j.err

set -euo pipefail
source /etc/profile
export SLURM_EXPORT_ENV=ALL

PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/IOSage
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
PYTHON="$IOSAGE_ENV/bin/python"
RESULTS_ROOT="$PROJECT_DIR/results/resubmission/iterative"
export PATH="$IOSAGE_ENV/bin:$PATH" PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$PROJECT_DIR"
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

: "${MODEL_BUNDLE:?set MODEL_BUNDLE to a final-evaluation bundle}"
: "${KNOWLEDGE_BASE:?set KNOWLEDGE_BASE to a schema 2 measured-evidence KB}"
[ -s "$MODEL_BUNDLE" ] || { echo "ERROR: model bundle missing or empty: $MODEL_BUNDLE" >&2; exit 1; }
[ -s "$KNOWLEDGE_BASE" ] || { echo "ERROR: knowledge base missing or empty: $KNOWLEDGE_BASE" >&2; exit 1; }
"$PYTHON" -c "import darshan, pandas, shap, xgboost; print('environment check passed')"

MODEL="${1:-claude-sonnet}"
if [ "$#" -ge 1 ]; then shift; fi
if [ "$#" -gt 0 ]; then
    WORKLOADS=("$@")
else
    WORKLOADS=(ior_small_posix ior_fsync_heavy ior_small_direct)
fi
N_RUNS="${N_RUNS:-3}"
case "$N_RUNS" in
    ''|*[!0-9]*|0) echo "ERROR: N_RUNS must be a positive integer" >&2; exit 2 ;;
esac
CONDITIONS=(full no_ml no_kb no_shap single_shot no_feedback)
RUN_TAG="${RUN_TAG:-ablation_${SLURM_JOB_ID:?SLURM_JOB_ID is required}}"
RUN_DIR="$RESULTS_ROOT/$RUN_TAG"
[ ! -e "$RUN_DIR" ] || { echo "ERROR: run directory already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR"

for CONDITION in "${CONDITIONS[@]}"; do
    case "$CONDITION" in
        full) EXTRA_ARGS=(); MAX_ITER=5; PREFIX=sweep ;;
        no_ml) EXTRA_ARGS=(--no-ml); MAX_ITER=5; PREFIX=ablation ;;
        no_kb) EXTRA_ARGS=(--no-kb); MAX_ITER=5; PREFIX=ablation ;;
        no_shap) EXTRA_ARGS=(--no-shap); MAX_ITER=5; PREFIX=ablation ;;
        single_shot) EXTRA_ARGS=(); MAX_ITER=1; PREFIX=ablation ;;
        no_feedback) EXTRA_ARGS=(--no-feedback); MAX_ITER=5; PREFIX=ablation ;;
    esac
    for WORKLOAD in "${WORKLOADS[@]}"; do
        OUTPUT="$RUN_DIR/${PREFIX}_${CONDITION}_${WORKLOAD}_${MODEL}.json"
        LOG="$RUN_DIR/${PREFIX}_${CONDITION}_${WORKLOAD}_${MODEL}.log"
        if ! "$PYTHON" -m src.llm.iterative_optimizer \
            --workload "$WORKLOAD" --model "$MODEL" --max-iterations "$MAX_ITER" \
            --n-runs "$N_RUNS" --model-bundle "$MODEL_BUNDLE" \
            --knowledge-base "$KNOWLEDGE_BASE" "${EXTRA_ARGS[@]}" \
            --output "$OUTPUT" >"$LOG" 2>&1; then
            tail -100 "$LOG" >&2
            exit 1
        fi
    done
done

"$PYTHON" scripts/aggregate_trackc_results.py \
    --results-dir "$RUN_DIR" --output "$RUN_DIR/summary.json"
echo "Validated ablation: $RUN_DIR"
