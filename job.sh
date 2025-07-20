#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH --job-name=ald_train
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-user=f20230424@pilani.bits-pilani.ac.in
#SBATCH --mail-type=END,FAIL

# (Optional) Load required module
module load nvhpc/22.3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ald_env

# Go to script directory
cd $SLURM_SUBMIT_DIR

# Run training
python ald_1.py
