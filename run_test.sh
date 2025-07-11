#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-annielee
#SBATCH --job-name=prep_data_test
#SBATCH --output=prep_data_test.txt
#SBATCH --mail-user=mjzhou@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

module load python gcc arrow
source ./myenv/bin/activate
python prep-pr-data.py

