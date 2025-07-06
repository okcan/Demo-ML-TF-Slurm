#!/bin/bash
#SBATCH --job-name=weather-train
#SBATCH --output=train_output.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=standard

module load python/3.9

cd /home/youruser/weather-forecasting
python3 model.py
