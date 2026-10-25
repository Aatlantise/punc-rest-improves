#!/bin/bash
#SBATCH --job-name="gpt2pr"
#SBATCH --account=def-annielee
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --mail-user=mjzhou@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python gcc arrow
source ../myenv/bin/activate

python train_gpt2.py