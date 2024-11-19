#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH -J cross_dataset
#SBATCH -o cross_dataset.log
#SBATCH -e cross_dataset.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

for seed in 42 53 64
do
    python -u cross_dataset.py --order lp_replace --seed ${seed}
    python -u cross_dataset.py --order replace_lp --seed ${seed}
done


