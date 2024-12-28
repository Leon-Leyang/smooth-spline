#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J post_cls_ablate_reg
#SBATCH -o post_cls_ablate_reg.log
#SBATCH -e post_cls_ablate_reg.log
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
  for reg in 0.1 10
  do
      python -u post_replace_classification.py --order lp_replace --reg ${reg} --seed ${seed}
      python -u post_replace_classification.py --order replace_lp --reg ${reg} --seed ${seed}
  done
done
