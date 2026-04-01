#!/bin/bash
# Clean SLURM submission wrapper.
# Unsets inherited SLURM env vars that cause conflicts on Delta
# when submitting child jobs from within a running SLURM job.
#
# Usage: bash scripts/sbatch_clean.sh scripts/run_e2e_pathological.slurm
#
# Problem: Parent job (e.g., Claude Code) sets SLURM_CPUS_PER_TASK=8,
# SLURM_TRES_PER_TASK=cpu=8. Child sbatch inherits these, causing
# "step creation disabled" errors when child uses --cpus-per-task=1.

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK \
      SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE \
      SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE \
      SLURM_CPU_BIND_VERBOSE SLURM_DISTRIBUTION \
      SLURM_JOB_CPUS_PER_NODE SLURM_NTASKS SLURM_NPROCS \
      SLURM_NNODES SLURM_NODELIST SLURM_JOB_NODELIST \
      SLURM_STEP_NODELIST SLURM_TASKS_PER_NODE \
      SLURM_JOB_NUM_NODES SLURM_STEP_NUM_TASKS \
      SLURM_STEP_NUM_NODES SLURM_STEP_TASKS_PER_NODE \
      2>/dev/null

sbatch "$@"
