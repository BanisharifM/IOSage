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
# =============================================================================
# Iterative: single-workload closed-loop optimization, driver runs as a batch job
# Usage: sbatch scripts/run_iterative_single.slurm [workload] [model] [max_iter] [n_runs]
#   e.g. sbatch scripts/run_iterative_single.slurm ior_small_posix claude-sonnet 5
# Environment: self-contained env on /work (conda explicit spec + pip freeze in configs/).
#   Override with: sbatch --export=NONE,IOSAGE_ENV=/path/to/env scripts/run_iterative_single.slurm
#   --export=NONE keeps the submitting shell out of the job; SLURM_EXPORT_ENV=ALL lets the
#   benchmark jobs this driver submits inherit the driver's (clean) environment.
# =============================================================================
export SLURM_EXPORT_ENV=ALL
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"

echo "============================================================"
echo "Iterative: Iterative LLM Code Optimization"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Env: $IOSAGE_ENV"
echo "============================================================"

# Environment: hermetic. PYTHONNOUSERSITE keeps ~/.local/lib/python3.9/site-packages out of
# the import path (the old /projects env silently depended on user-site xgboost/shap/lightgbm).
export PATH="$IOSAGE_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

cd /work/hdd/bdau/mbanisharifdehkordi/IOSage || exit 1

# API key (set in environment or .env file)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Default args, can be overridden via command line
WORKLOAD="${1:-ior_small_posix}"
MODEL="${2:-claude-sonnet}"
MAX_ITER="${3:-5}"
N_RUNS="${4:-1}"

echo "Workload: $WORKLOAD"
echo "Model: $MODEL"
echo "Max iterations: $MAX_ITER"
echo "Runs: $N_RUNS"
"$IOSAGE_ENV/bin/python" -c "import xgboost, shap, darshan, pandas; print('env check ok: xgboost', xgboost.__version__)" || exit 1
echo ""

# Run
"$IOSAGE_ENV/bin/python" -m src.llm.iterative_optimizer \
    --workload "$WORKLOAD" \
    --model "$MODEL" \
    --max-iterations "$MAX_ITER" \
    --n-runs "$N_RUNS" \
    --output "results/resubmission/iterative/iter_${WORKLOAD}_${MODEL}_$(date +%s).json"

echo ""
echo "Completed: $(date)"
