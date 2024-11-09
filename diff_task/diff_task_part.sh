#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH -J diff_task_part
#SBATCH -o diff_task_part.log
#SBATCH -e diff_task_part.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

config=diff_task/voc2012_pspnet50.yaml

for beta in $(seq 0.95 0.01 1.00)
do
    python -u diff_task/diff_task_part.py --config=${config} --beta ${beta}
done
