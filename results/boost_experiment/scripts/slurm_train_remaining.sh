#!/bin/bash
#SBATCH --job-name=train_all_models
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/train_remaining_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/logs/train_remaining_%j.err

/projects/bdau/envs/sc2026/bin/python /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/boost_experiment/scripts/train_remaining_models.py
