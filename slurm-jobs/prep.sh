#!/bin/bash
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:0:0
module load python gcc arrow
source .env/bin/activate
python prep-pr-data.py
deactivate
