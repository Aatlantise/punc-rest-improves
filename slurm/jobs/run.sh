#!/bin/bash

#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=1
#COMMENT --gpus=1
#SBATCH --job-name='data-prep'
#SBATCH --mail-user='rex.fang@icloud.com'
#SBATCH --mem=16G
#SBATCH --output="slurm/logs/%x-%j"
#SBATCH --time=1:0:0

source env.sh
python prep_data/wiki.py
python prep_data/conll_2012.py
