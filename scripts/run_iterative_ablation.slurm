#!/bin/bash
#SBATCH --job-name=iter_ablation
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/iterative/ablation_%j.out
#SBATCH --error=results/iterative/ablation_%j.err

# =============================================================================
# Iterative: Ablation study (6 conditions x 4 representative workloads x Claude)
# =============================================================================

echo "============================================================"
echo "Iterative: ABLATION STUDY"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:$PATH
export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:$PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

if [ -f .env ]; then
    source .env
fi

TIMESTAMP=$(date +%s)
MODEL=claude-sonnet
N_RUNS=3
MAX_ITER=5

# Representative workloads (1 per bottleneck type)
WORKLOADS="ior_small_posix ior_fsync_heavy ior_random_access mdtest_metadata_storm"

# A0: Full system (already in sweep, skip here)
echo "A0: Full system -- see sweep results"

# A1: No ML
echo ""
echo "========== ABLATION A1: No ML =========="
for W in $WORKLOADS; do
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --workload "$W" --model $MODEL --max-iterations $MAX_ITER --n-runs $N_RUNS \
        --no-ml \
        --output "results/iterative/ablation_no_ml_${W}_${TIMESTAMP}.json"
done

# A2: No KB
echo ""
echo "========== ABLATION A2: No KB =========="
for W in $WORKLOADS; do
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --workload "$W" --model $MODEL --max-iterations $MAX_ITER --n-runs $N_RUNS \
        --no-kb \
        --output "results/iterative/ablation_no_kb_${W}_${TIMESTAMP}.json"
done

# A3: No SHAP
echo ""
echo "========== ABLATION A3: No SHAP =========="
for W in $WORKLOADS; do
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --workload "$W" --model $MODEL --max-iterations $MAX_ITER --n-runs $N_RUNS \
        --no-shap \
        --output "results/iterative/ablation_no_shap_${W}_${TIMESTAMP}.json"
done

# A4: Single-shot (1 iteration = Single-shot equivalent)
echo ""
echo "========== ABLATION A4: Single-shot =========="
for W in $WORKLOADS; do
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --workload "$W" --model $MODEL --max-iterations 1 --n-runs $N_RUNS \
        --output "results/iterative/ablation_single_shot_${W}_${TIMESTAMP}.json"
done

# A5: No feedback
echo ""
echo "========== ABLATION A5: No feedback =========="
for W in $WORKLOADS; do
    /projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
        --workload "$W" --model $MODEL --max-iterations $MAX_ITER --n-runs $N_RUNS \
        --no-feedback \
        --output "results/iterative/ablation_no_feedback_${W}_${TIMESTAMP}.json"
done

echo ""
echo "============================================================"
echo "ABLATION STUDY COMPLETED: $(date)"
echo "============================================================"
