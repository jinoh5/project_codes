#!/bin/bash

#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --time=5:00:00
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yujinoh@uchicago.edu

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hw3

# Debugging: Check if environment is activated
echo "Current Python: $(which python)"
echo "Conda Environments:"

# Do the work
cd /home/yujinoh/temp_project
python run_gpt2.py