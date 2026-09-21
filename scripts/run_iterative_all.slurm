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
# =============================================================================
# Run the closed loop over several workloads, one after another.
# Sequential on purpose: the objective is wall time, so two benchmark jobs must not
# share the filesystem while either is being timed.
# Usage: sbatch scripts/run_iterative_all.slurm [model] [max_iter] [workload ...]
#   default model claude-sonnet, 5 iterations, every workload in configs/iterative.yaml
#   except ones already measured (pass workload names explicitly to override).
# =============================================================================
export SLURM_EXPORT_ENV=ALL
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
export PATH="$IOSAGE_ENV/bin:$PATH" PYTHONNOUSERSITE=1
unset PYTHONPATH
cd /work/hdd/bdau/mbanisharifdehkordi/IOSage || exit 1
[ -f .env ] && { set -a; source .env; set +a; }

MODEL="${1:-claude-sonnet}"; MAX_ITER="${2:-5}"; shift 2 2>/dev/null
WORKLOADS="$*"
if [ -z "$WORKLOADS" ]; then
    WORKLOADS=$("$IOSAGE_ENV/bin/python" -c "
import yaml
done = {'ior_small_posix'}
print(' '.join(k for k in yaml.safe_load(open('configs/iterative.yaml'))['workloads'] if k not in done))")
fi

echo "host $(hostname)  model $MODEL  iterations $MAX_ITER"
echo "workloads: $WORKLOADS"
"$IOSAGE_ENV/bin/python" -c "import xgboost, shap, openai; print('env check ok')" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="results/resubmission/iterative/sweep_${MODEL}_${STAMP}_summary.tsv"
printf "workload\tbest_speedup\tstatus\tcost_usd\tlog\n" > "$SUMMARY"

for W in $WORKLOADS; do
    LOG="results/resubmission/iterative/sweep_${STAMP}_${W}_${MODEL}.log"
    echo "=== $(date +%H:%M:%S)  $W"
    OUT="results/resubmission/iterative/iter_${W}_${MODEL}_$(date +%s).json"
    "$IOSAGE_ENV/bin/python" -m src.llm.iterative_optimizer \
        --workload "$W" --model "$MODEL" --max-iterations "$MAX_ITER" \
        --n-runs 1 --output "$OUT" > "$LOG" 2>&1
    rc=$?
    "$IOSAGE_ENV/bin/python" - "$W" "$OUT" "$rc" "$LOG" >> "$SUMMARY" <<'PY'
import json, sys
w, out, rc, log = sys.argv[1:5]
try:
    d = json.load(open(out)); h = d[0] if isinstance(d, list) else d
    print(f"{w}\t{h.get('best_speedup')}\t{h.get('final_status')}\t{h.get('total_cost_usd')}\t{log}")
except Exception as e:
    print(f"{w}\tNA\tdriver_rc={rc} ({type(e).__name__})\tNA\t{log}")
PY
    echo "    -> $(tail -1 "$SUMMARY")"
done

echo "=== finished $(date)"
column -t -s $'\t' "$SUMMARY"
