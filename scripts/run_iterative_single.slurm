#!/bin/bash
#SBATCH --job-name=iter_optimize
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/iterative/orchestrator_%j.out
#SBATCH --error=results/iterative/orchestrator_%j.err

# =============================================================================
# Track C: Single workload iterative optimization test
# Usage: sbatch scripts/run_iterative_single.slurm
# Or with args: sbatch scripts/run_iterative_single.slurm --workload ior_small_posix --model claude-sonnet
# =============================================================================

echo "============================================================"
echo "Track C: Iterative LLM Code Optimization"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Environment
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

# API key (set in environment or .env file)
if [ -f .env ]; then
    source .env
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
echo ""

# Run
/projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
    --workload "$WORKLOAD" \
    --model "$MODEL" \
    --max-iterations "$MAX_ITER" \
    --n-runs "$N_RUNS" \
    --output "results/iterative/iter_${WORKLOAD}_${MODEL}_$(date +%s).json"

echo ""
echo "Completed: $(date)"
