#!/bin/bash
# E2E NetCDF-4 write kernels on Delta: Cray PE defaults (PrgEnv-gnu, cray-mpich) with parallel
# HDF5 and NetCDF; CPU-only; Anaconda stripped from PATH/LD_LIBRARY_PATH (ld resolves
# shared-library dependencies through LD_LIBRARY_PATH).
source /etc/profile.d/modules.sh
module reset
module unload craype-accel-nvidia80 cudatoolkit 2>/dev/null
module load cray-hdf5-parallel/1.14.3.9 cray-netcdf-hdf5parallel/4.9.2.3
if type conda >/dev/null 2>&1; then conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; fi
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE '/anaconda3/|/miniconda3/|/\.conda/' | paste -sd: -)
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '/anaconda3/|/miniconda3/|/\.conda/' | paste -sd: -)
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL CMAKE_PREFIX_PATH MKLROOT
