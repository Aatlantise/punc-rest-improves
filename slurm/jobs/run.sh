#!/bin/bash

#SBATCH -c 4
#SBATCH -G 1
#SBATCH -t 12:0:0

#SBATCH --job-name='srl-train'
#SBATCH --mail-user='rex.fang@icloud.com'
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#SBATCH --output="slurm/logs/%x-%j"

module load arrow cuda gcc python
source .env/bin/activate # needs to be run with source, not bash
pip install -r requirements.txt # make sure all requirements are satisfied 
export TOKENIZERS_PARALLELISM=true
python train.py
