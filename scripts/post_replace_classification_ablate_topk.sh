#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J post_cls_ablate_topk
#SBATCH -o post_cls_ablate_topk.log
#SBATCH -e post_cls_ablate_topk.log
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
  for topk in 2 3
  do
      python -u post_replace_classification.py --order replace_lp --topk ${topk} --seed ${seed}
  done
done
