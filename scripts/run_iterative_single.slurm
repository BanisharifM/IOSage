#!/bin/bash
#SBATCH --job-name=iter_optimize
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/orchestrator_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/iterative/orchestrator_%j.err

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

WORKLOAD="${1:-ior_small_posix}"
MODEL="${2:-claude-sonnet}"
MAX_ITER="${3:-5}"
N_RUNS="${4:-1}"
CONDITION="${5:-full}"
case "$CONDITION" in
    full) EXTRA_ARGS=(); PREFIX=iter ;;
    no_ml) EXTRA_ARGS=(--no-ml); PREFIX=ablation ;;
    no_kb) EXTRA_ARGS=(--no-kb); PREFIX=ablation ;;
    no_shap) EXTRA_ARGS=(--no-shap); PREFIX=ablation ;;
    no_feedback) EXTRA_ARGS=(--no-feedback); PREFIX=ablation ;;
    single_shot) EXTRA_ARGS=(); PREFIX=ablation; MAX_ITER=1 ;;
    *) echo "ERROR: unsupported condition: $CONDITION" >&2; exit 2 ;;
esac
case "$MAX_ITER:$N_RUNS" in
    *[!0-9:]*|0:*|*:0) echo "ERROR: iteration and run counts must be positive integers" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-single_${SLURM_JOB_ID:?SLURM_JOB_ID is required}}"
RUN_DIR="$RESULTS_ROOT/$RUN_TAG"
[ ! -e "$RUN_DIR" ] || { echo "ERROR: run directory already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR"
OUTPUT="$RUN_DIR/${PREFIX}_${CONDITION}_${WORKLOAD}_${MODEL}.json"

"$PYTHON" -m src.llm.iterative_optimizer \
    --workload "$WORKLOAD" --model "$MODEL" --max-iterations "$MAX_ITER" \
    --n-runs "$N_RUNS" --model-bundle "$MODEL_BUNDLE" \
    --knowledge-base "$KNOWLEDGE_BASE" "${EXTRA_ARGS[@]}" --output "$OUTPUT"
"$PYTHON" - "$RUN_DIR" <<'PY'
import sys
from src.llm.iterative_result import load_iterative_results
records = load_iterative_results(sys.argv[1])
print(f"validated {len(records)} iterative result record(s)")
PY
