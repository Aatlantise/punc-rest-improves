#!/bin/bash
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:0:0
module load arrow gcc python
source .env/bin/activate
python main.py train punct_restore_dataset_20231101.fr.jsonl -m xlm-mlm-enfr-1024
deactivate
