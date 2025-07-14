#!/bin/bash
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=30:00
#SBATCH --mail-user=mjzhou@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python gcc arrow
source ./myenv/bin/activate
python main_fine_tune_conll.py --model_name_or_path trained_PR_model --output_dir outputs --NER_eval_flag
