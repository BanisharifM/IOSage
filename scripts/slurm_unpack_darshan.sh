#!/bin/bash
# =============================================================================
# SLURM Job: Unpack Darshan Logs
# =============================================================================
# Submits a CPU job to unpack all logs.tar.gz files in the Darshan collection.
#
# Submit with:
#   sbatch scripts/slurm_unpack_darshan.sh
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/slurm/unpack_darshan_*.out
# =============================================================================

#SBATCH --job-name=unpack_darshan
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/unpack_darshan_%j.out
#SBATCH --error=logs/slurm/unpack_darshan_%j.err

echo "=== Unpack Darshan Logs ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Start:     $(date)"
echo ""

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026

# Unpack all tarballs with 32 parallel workers
bash scripts/unpack_darshan_logs.sh --workers 32

echo ""
echo "=== Disk Usage ==="
du -sh Darshan_Logs/
du -sh Darshan_Logs/2024/ Darshan_Logs/2025/ Darshan_Logs/2026/ 2>/dev/null

echo ""
echo "=== Done ==="
echo "End: $(date)"
