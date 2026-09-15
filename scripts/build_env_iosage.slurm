#!/bin/bash
#SBATCH --job-name=build_env
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/iterative/build_env_%j.out
# =============================================================================
# Rebuild the paper's Python environment on /work from the captured specs
# (configs/sc2026_explicit_spec.txt: conda explicit URLs; configs/sc2026_pip_freeze.txt: pip).
# Self-contained: package caches on /work, no /projects (Taiga) dependency, no user site.
# Usage: sbatch [--exclude=<bad node>] scripts/build_env_iosage.slurm
# =============================================================================
set -u
REPO=/work/hdd/bdau/mbanisharifdehkordi/IOSage
ENVS="${ENVS:-/work/hdd/bdau/mbanisharifdehkordi/envs}"   # override: sbatch --export=NONE,ENVS=/work/nvme/...
PREFIX=$ENVS/iosage
mkdir -p "$ENVS"
export CONDARC=$ENVS/condarc
[ -f "$CONDARC" ] || printf "solver: libmamba\nauto_activate_base: false\nenvs_dirs:\n  - %s\npkgs_dirs:\n  - %s/.conda_pkgs\n" "$ENVS" "$ENVS" > "$CONDARC"
export CONDA_PKGS_DIRS=$ENVS/.conda_pkgs
export PIP_CACHE_DIR=$ENVS/.pip_cache
export PYTHONNOUSERSITE=1      # pip must not see ~/.local packages as already satisfied
echo "host $(hostname) start $(date)"
if [ -e "$PREFIX" ]; then echo "prefix exists: $PREFIX (refusing to overwrite)"; exit 2; fi
CONDA=/sw/external/python/anaconda3/bin/conda
$CONDA create -y -p "$PREFIX" --file "$REPO/configs/sc2026_explicit_spec.txt"
rc=$?; echo "conda create rc=$rc"; [ $rc -ne 0 ] && { echo "ENV BUILD rc=$rc"; exit $rc; }
"$PREFIX/bin/python" -m pip install -r "$REPO/configs/sc2026_pip_freeze.txt"
rc=$?; echo "pip install rc=$rc"
"$PREFIX/bin/python" -c "import xgboost, shap, lightgbm, pandas, darshan, sklearn, numpy; print('imports ok: xgboost', xgboost.__version__, 'shap', shap.__version__, 'numpy', numpy.__version__); import xgboost, os; print('xgboost from', os.path.dirname(xgboost.__file__))"
rc2=$?
echo "files: $(find "$PREFIX" -type f | wc -l)  size: $(du -sh "$PREFIX" | cut -f1)"
echo "end $(date)"
echo "ENV BUILD rc=$((rc + rc2))"
