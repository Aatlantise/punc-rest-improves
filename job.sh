#!/bin/bash
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:0:0
module load python gcc arrow
source .env/bin/activate
python main.py
deactivate
