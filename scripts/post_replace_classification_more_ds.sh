#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J post_cls_more_ds
#SBATCH -o post_cls_more_ds.log
#SBATCH -e post_cls_more_ds.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

for seed in 42 43 44
do
    python -u post_replace_classification.py --order replace_lp --seed ${seed} --pretrained_ds imagenet --transfer_ds arabic_characters fgvc_aircraft flowers102 fashion_mnist med_mnist/pathmnist arabic_digits beans cub200 dtd food101 med_mnist/chestmnist med_mnist/dermamnist
done


