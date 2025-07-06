#!/bin/bash
#SBATCH --job-name=weather-train
#SBATCH --output=train_output.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --partition=debug

module load python/3.9

cd cd /home/yasinsaygili/demo/Demo-ML-TF-Slurm/Demo-ML-TF-Slurm
python3 model.py

