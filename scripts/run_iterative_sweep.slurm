#!/bin/bash
#SBATCH --job-name=iter_sweep
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/iterative/sweep_%j.out
#SBATCH --error=results/iterative/sweep_%j.err

# =============================================================================
# Iterative: Full iterative optimization sweep
# Runs all workloads x all models x 5 repetitions
# Estimated: ~8 workloads x 3 models x 5 runs x 3 avg iters = ~360 benchmark jobs
# =============================================================================

echo "============================================================"
echo "Iterative: FULL SWEEP"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

if [ -f .env ]; then
    source .env
fi

TIMESTAMP=$(date +%s)
N_RUNS=5
MAX_ITER=5

# Run each model sequentially (to avoid API rate limits)
for MODEL in claude-sonnet gpt-4o llama-70b; do
    echo ""
    echo "========== MODEL: $MODEL =========="
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --sweep \
        --model "$MODEL" \
        --max-iterations $MAX_ITER \
        --n-runs $N_RUNS \
        --output "results/iterative/sweep_${MODEL}_${TIMESTAMP}.json"
    echo "Model $MODEL completed at $(date)"
done

echo ""
echo "============================================================"
echo "FULL SWEEP COMPLETED: $(date)"
echo "============================================================"
