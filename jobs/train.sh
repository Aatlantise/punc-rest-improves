#!/bin/bash
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:0:0
module load python gcc arrow
source .env/bin/activate
python main.py
deactivate
