#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -p cs-all-gcondo --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH -J cross_dataset_ablate_C
#SBATCH -o cross_dataset_ablate_C.log
#SBATCH -e cross_dataset_ablate_C.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

for C in 0.01 100
do
    python -u cross_dataset.py --order lp_replace --C ${C}
    python -u cross_dataset.py --order replace_lp --C ${C}
done


