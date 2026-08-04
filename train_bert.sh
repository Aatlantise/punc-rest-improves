#!/bin/bash
#SBATCH --job-name="bertpr2"
#SBATCH --account=def-annielee
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --time=18:57:0
#SBATCH --mail-user=mjzhou@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python gcc arrow
source ../myenv/bin/activate

python train_bert.py >> train_bert.log