#!/bin/bash

#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --job-name="srl-training-attempt"
#SBATCH --mem=16G
#SBATCH --output="slurm-logs/%x-%j"
#SBATCH --time=10:00

source env.sh
python train.py
