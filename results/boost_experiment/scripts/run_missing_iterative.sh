#!/bin/bash
# Submit missing iterative workloads for GPT-4o and Llama-70b
# These fill in the "–" cells in Table VII
#
# Missing workloads for BOTH GPT-4o and Llama:
#   DLIO: dlio_small_records, dlio_many_small_files, dlio_shuffle_heavy
#   h5bench: h5bench_small_access, h5bench_indep_vs_coll, h5bench_interleaved_pattern
#   HACC-IO: hacc_posix_shared_small, hacc_fpp_small
#   custom: custom_load_imbalance

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

echo "=== Submitting missing iterative workloads ==="
echo "Workloads: ${#WORKLOADS[@]}"
echo "Models: ${#MODELS[@]}"
echo "Total jobs: $((${#WORKLOADS[@]} * ${#MODELS[@]}))"
echo ""

for model in "${MODELS[@]}"; do
    for wl in "${WORKLOADS[@]}"; do
        echo "Submitting: $wl / $model"
        sbatch --job-name="iter_${wl}_${model}" \
               --partition=cpu \
               --account=bdau-delta-cpu \
               --nodes=1 \
               --ntasks=1 \
               --cpus-per-task=4 \
               --mem=32G \
               --time=04:00:00 \
               --output="results/iterative/orchestrator_${wl}_${model}_%j.out" \
               --error="results/iterative/orchestrator_${wl}_${model}_%j.err" \
               --wrap="
source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:\$PATH
export PYTHONPATH=/work/hdd/bdau/mbanisharifdehkordi/.local_pkgs:\$PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
source .env
/projects/bdau/envs/sc2026/bin/python -m src.llm.iterative_optimizer \
    --workload '$wl' \
    --model '$model' \
    --max-iterations 5 \
    --n-runs 1 \
    --output 'results/iterative/trackc_${wl}_${model}.json'
"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
