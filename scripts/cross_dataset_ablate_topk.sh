#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH -J cross_dataset_ablate_topk
#SBATCH -o cross_dataset_ablate_topk.log
#SBATCH -e cross_dataset_ablate_topk.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

for topk in 2 3
do
    python -u cross_dataset.py --order lp_replace --topk ${topk}
    python -u cross_dataset.py --order replace_lp --topk ${topk}
done


