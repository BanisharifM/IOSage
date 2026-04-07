#!/bin/bash
#SBATCH --job-name=e2e_boost_analysis
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/e2e_analysis_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/e2e_analysis_%j.err

cd /work/hdd/bdau/mbanisharifdehkordi/SC_2026
source .env

/projects/bdau/envs/sc2026/bin/python results/boost_experiment/scripts/run_e2e_pipeline_analysis.py

echo "Exit code: $?"
