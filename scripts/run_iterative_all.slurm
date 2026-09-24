#!/bin/bash
#SBATCH --job-name=iter_all
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/iter_all_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/iter_all_%j.err

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
MAX_ITER="${2:-5}"
if [ "$#" -ge 2 ]; then shift 2; else set --; fi
case "$MAX_ITER" in
    ''|*[!0-9]*|0) echo "ERROR: max iterations must be a positive integer" >&2; exit 2 ;;
esac
if [ "$#" -gt 0 ]; then
    WORKLOADS=("$@")
else
    mapfile -t WORKLOADS < <("$PYTHON" - <<'PY'
import yaml
with open("configs/iterative.yaml") as handle:
    workloads = yaml.safe_load(handle)["workloads"]
    print("\n".join(
        name for name, config in workloads.items()
        if config.get("classifier_supported", True)
    ))
PY
)
fi
[ "${#WORKLOADS[@]}" -gt 0 ] || { echo "ERROR: no workloads selected" >&2; exit 2; }

RUN_TAG="${RUN_TAG:-sweep_${SLURM_JOB_ID:?SLURM_JOB_ID is required}}"
RUN_DIR="$RESULTS_ROOT/$RUN_TAG"
[ ! -e "$RUN_DIR" ] || { echo "ERROR: run directory already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR"

for WORKLOAD in "${WORKLOADS[@]}"; do
    OUTPUT="$RUN_DIR/sweep_${WORKLOAD}_${MODEL}.json"
    LOG="$RUN_DIR/sweep_${WORKLOAD}_${MODEL}.log"
    if ! "$PYTHON" -m src.llm.iterative_optimizer \
        --workload "$WORKLOAD" --model "$MODEL" --max-iterations "$MAX_ITER" \
        --n-runs 1 --model-bundle "$MODEL_BUNDLE" \
        --knowledge-base "$KNOWLEDGE_BASE" --output "$OUTPUT" >"$LOG" 2>&1; then
        tail -100 "$LOG" >&2
        exit 1
    fi
done

"$PYTHON" scripts/aggregate_trackc_results.py \
    --results-dir "$RUN_DIR" --output "$RUN_DIR/summary.json"
echo "Validated sweep: $RUN_DIR"
