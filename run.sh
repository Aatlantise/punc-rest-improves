#!/bin/bash
#SBATCH --job-name="pr"
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --output="%x.o%j"
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL
#############################################################

# install the environment by loading in python and required packages
source env.sh

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
python main.py >> train.log
python eval.py >> eval.log
