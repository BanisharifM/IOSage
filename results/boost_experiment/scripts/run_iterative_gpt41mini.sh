#!/bin/bash
# Run ALL iterative workloads with GPT-4.1-mini using the BOOST EXPERIMENT model
# Copied from run_all_iterative_boost.sh, only model changed
# Results saved to results/boost_experiment/full_evaluation/iterative/

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

OUTDIR="results/boost_experiment/full_evaluation/iterative"
mkdir -p "$OUTDIR"

WORKLOADS=(
    "ior_small_posix"
    "ior_small_direct"
    "ior_fsync_heavy"
    "ior_random_access"
    "ior_misaligned"
    "ior_healthy_baseline"
    "ior_interface_shared"
    "mdtest_metadata_storm"
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

MODEL="gpt-4.1-mini"
MNAME="gpt41mini"

echo "=== Submitting ALL iterative workloads with GPT-4.1-mini + BOOST model ==="
echo "Workloads: ${#WORKLOADS[@]}"
echo "Model: $MODEL"
echo "Output: $OUTDIR"
echo ""

for wl in "${WORKLOADS[@]}"; do
    SCRIPT=$(mktemp /tmp/iter_boost_XXXXXX.sh)
    cat > "$SCRIPT" << ENDSCRIPT
#!/bin/bash
#SBATCH --job-name=boost_${wl}_${MNAME}
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=${OUTDIR}/orchestrator_${wl}_${MNAME}_%j.out
#SBATCH --error=${OUTDIR}/orchestrator_${wl}_${MNAME}_%j.err

source /projects/bdau/envs/sc2026/bin/activate 2>/dev/null || true
export PATH=/projects/bdau/envs/sc2026/bin:\$PATH
unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
source .env

echo "Workload: ${wl}"
echo "Model: ${MODEL}"
echo "Using BOOST model"
echo "Start: \$(date)"

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/scripts/iterative_optimizer_boost.py \
    --workload "${wl}" \
    --model "${MODEL}" \
    --max-iterations 5 \
    --n-runs 1 \
    --no-shap \
    --output "${OUTDIR}/trackc_${wl}_${MNAME}.json"

echo "End: \$(date)"
ENDSCRIPT
    echo "Submitting: $wl / $MODEL"
    sbatch "$SCRIPT"
    rm "$SCRIPT"
done

echo ""
echo "All ${#WORKLOADS[@]} jobs submitted."
echo "Results will be in: $OUTDIR"
