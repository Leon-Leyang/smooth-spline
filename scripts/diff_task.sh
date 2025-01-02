#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH -J diff_task
#SBATCH -o diff_task.log
#SBATCH -e diff_task.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

config=voc2012_pspnet50.yaml

for seed in 42 43 44
do
    for beta in $(seq 0.5 0.01 1)
    do
        python -u diff_task.py --config=${config} --beta ${beta} manual_seed ${seed}
    done
done
