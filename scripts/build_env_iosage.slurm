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
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/build_env_%j.out
# =============================================================================
# Rebuild the paper's Python environment on /work from the captured specs
# (configs/sc2026_explicit_spec.txt: conda explicit URLs; configs/sc2026_pip_freeze.txt: pip).
# Self-contained: package caches on /work, no /projects (Taiga) dependency, no user site.
# Usage: sbatch [--exclude=<bad node>] scripts/build_env_iosage.slurm
# =============================================================================
set -euo pipefail
source /etc/profile
REPO=/work/hdd/bdau/mbanisharifdehkordi/IOSage
ENVS="${ENVS:-/work/nvme/bdau/mbanisharifdehkordi/envs}"   # /work/nvme: small-file creates ~50x faster than /work/hdd
PREFIX=$ENVS/iosage
DARSHAN_SRC_DIR="${DARSHAN_SRC_DIR:-/work/hdd/bdau/mbanisharifdehkordi/IOSage_runtime/tools/src}"
mkdir -p "$ENVS" "$DARSHAN_SRC_DIR"
export CONDARC=$ENVS/condarc
[ -f "$CONDARC" ] || printf "solver: libmamba\nauto_activate_base: false\nenvs_dirs:\n  - %s\npkgs_dirs:\n  - %s/.conda_pkgs\n" "$ENVS" "$ENVS" > "$CONDARC"
export CONDA_PKGS_DIRS=$ENVS/.conda_pkgs
export PIP_CACHE_DIR=$ENVS/.pip_cache
export PYTHONNOUSERSITE=1      # pip must not see ~/.local packages as already satisfied
echo "host $(hostname) start $(date)"
CONDA=/sw/external/python/anaconda3/bin/conda
# Resume: an explicit spec is fully determined, so a prefix whose conda-meta lists exactly the
# spec's packages is the finished conda layer; anything else is refused rather than overwritten.
n_spec=$(grep -c -E '^https?://' "$REPO/configs/sc2026_explicit_spec.txt")
if [ -e "$PREFIX" ]; then
    shopt -s nullglob
    meta_files=("$PREFIX"/conda-meta/*.json)
    n_meta=${#meta_files[@]}
    shopt -u nullglob
    if [ "$n_meta" -eq "$n_spec" ]; then echo "conda layer present ($n_meta/$n_spec packages), resuming at pip"
    else echo "prefix exists with $n_meta/$n_spec conda packages: $PREFIX (refusing to overwrite)"; exit 2; fi
else
    $CONDA create -y -p "$PREFIX" --file "$REPO/configs/sc2026_explicit_spec.txt"
    echo "conda create rc=0"
fi
# darshan-util 3.5.0 with bzip2 into the env prefix (the Polaris logs are bzip2 compressed; the
# PyDarshan wheel's bundled library has no bzip2, so the freeze installs PyDarshan from source and
# it loads this library through the RPATH of _cffi_backend, $ORIGIN/../.. = $PREFIX/lib).
DARSHAN_SRC=$DARSHAN_SRC_DIR/darshan-3.5.0
if [ ! -f "$DARSHAN_SRC.tar.gz" ]; then
    curl -fsSL -o "$DARSHAN_SRC.tar.gz" https://github.com/darshan-hpc/darshan/releases/download/3.5.0/darshan-3.5.0.tar.gz
fi
echo "5299ae5407ef55f4503bfaa038cf8b01128d19238b10e3e20d980411c2e8b97c  $DARSHAN_SRC.tar.gz" | sha256sum -c - || { echo "ENV BUILD: darshan tarball checksum mismatch"; exit 3; }
[ -d "$DARSHAN_SRC" ] || tar xzf "$DARSHAN_SRC.tar.gz" -C "$DARSHAN_SRC_DIR"
mkdir -p "$DARSHAN_SRC/darshan-util/build"
cd "$DARSHAN_SRC/darshan-util/build"
../configure --prefix="$PREFIX" --with-zlib=/usr --with-bzlib=/usr CC=gcc > configure.log 2>&1 && make -j4 > make.log 2>&1 && make install > install.log 2>&1
echo "darshan-util build rc=0"
grep -q "HAVE_LIBBZ2 1" darshan-util-config.h || { echo "ENV BUILD: darshan-util configured without bzip2"; exit 3; }
cd "$REPO"
"$PREFIX/bin/python" -m pip install -r "$REPO/configs/sc2026_pip_freeze.txt"
echo "pip install rc=0"
"$PREFIX/bin/python" -m pip check
echo "pip check rc=0"
"$PREFIX/bin/python" "$REPO/scripts/check_shared_library_owners.py"
echo "shared-library owner check rc=0"
"$PREFIX/bin/python" -c "import xgboost, shap, lightgbm, pandas, darshan, sklearn, numpy, openai, anthropic; print('imports ok: xgboost', xgboost.__version__, 'shap', shap.__version__, 'numpy', numpy.__version__, 'openai', openai.__version__); import xgboost, os; print('xgboost from', os.path.dirname(xgboost.__file__))"
"$PREFIX/bin/python" -c "
import darshan
from darshan.backend.cffi_backend import libdutil, ffi
lib = [l.split()[-1] for l in open('/proc/self/maps') if 'libdarshan-util' in l][0]
assert lib.startswith('$PREFIX/lib/'), lib
print('libdarshan-util', ffi.string(libdutil.darshan_log_get_lib_version()).decode(), 'from', lib)"
echo "darshan library check rc=0"
echo "files: $(find "$PREFIX" -type f | wc -l)  size: $(du -sh "$PREFIX" | cut -f1)"
echo "end $(date)"
echo "ENV BUILD rc=0"
