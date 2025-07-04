#!/bin/bash

#SBATCH --job-name="pr"
#SBATCH --output="%x.o%j"
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=def-annielee
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python prep-pr-data.py
python main.py