#!/bin/bash
# Submit missing iterative workloads for GPT-4o and Llama-70b
# Fixed: use heredoc instead of --wrap to avoid quoting issues

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

WORKLOADS=(
    "dlio_small_records"
    "dlio_many_small_files"
    "dlio_shuffle_heavy"
    "h5bench_small_access"
    "h5bench_indep_vs_coll"
    "h5bench_interleaved_pattern"
    "hacc_posix_shared_small"
    "hacc_fpp_small"
    "custom_load_imbalance"
)

MODELS=("gpt-4o" "llama-70b")

echo "=== Submitting missing iterative workloads (v2) ==="

for model in "${MODELS[@]}"; do
    for wl in "${WORKLOADS[@]}"; do
        SCRIPT=$(mktemp /tmp/iter_XXXXXX.sh)
        cat > "$SCRIPT" << ENDSCRIPT
#!/bin/bash
#SBATCH --job-name=iter_${wl}_${model}
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/iterative/orchestrator_${wl}_${model}_%j.out
#SBATCH --error=results/iterative/orchestrator_${wl}_${model}_%j.err

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:\$PATH
# Do NOT add .local_pkgs to PYTHONPATH — it contains numpy 1.24.3 which
# shadows the conda env's numpy 1.26.4 and breaks TensorFlow/DLIO imports
# (AttributeError: module 'numpy' has no attribute 'dtypes')
unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
source .env

echo "Workload: ${wl}"
echo "Model: ${model}"
echo "Start: \$(date)"

/projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
    --workload "${wl}" \
    --model "${model}" \
    --max-iterations 5 \
    --n-runs 1 \
    --output "results/iterative/trackc_${wl}_${model}.json"

echo "End: \$(date)"
ENDSCRIPT
        echo "Submitting: $wl / $model"
        sbatch "$SCRIPT"
        rm "$SCRIPT"
    done
done

echo ""
echo "All jobs submitted."
