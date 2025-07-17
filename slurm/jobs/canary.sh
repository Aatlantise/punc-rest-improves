#!/bin/bash

#SBATCH -c 4
#COMMENT -G 1
#SBATCH -t 10:00

#SBATCH --job-name='srl-train-canary'
#SBATCH --mem=16G
#SBATCH --output="slurm/logs/%x-%j"

module load arrow cuda gcc python
source .env/bin/activate # needs to be run with source, not bash
pip install -r requirements.txt # make sure all requirements are satisfied 
export TOKENIZERS_PARALLELISM=true
python train.py
