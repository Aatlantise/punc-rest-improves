#!/bin/bash

#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
#SBATCH --job-name="testing-conll-2012-data-prep"
#SBATCH --mail-user=h39fang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=8G
#SBATCH --output="%x.o%j"
#SBATCH --time=5:00

source env.sh
pip install -r requirements.txt
python prep_data/conll_2012.py --trust_remote_code
