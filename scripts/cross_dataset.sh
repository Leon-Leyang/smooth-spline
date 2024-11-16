#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
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

for seed in 42 123 456
do
    python -u diff_dataset.py --order lp_replace --seed ${seed}
    python -u diff_dataset.py --order replace_lp --seed ${seed}
    python -u same_dataset.py --seed ${seed}
done


