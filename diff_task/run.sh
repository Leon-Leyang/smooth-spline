#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH -J train
#SBATCH -o train.log
#SBATCH -e train.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leyang_hu@brown.edu

# Load the necessary module
module load miniconda3/23.11.0s

eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate spline

export PYTHONPATH=./

dataset=voc2012
exp_name=pspnet50
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=diff_task/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp diff_task/run.sh diff_task/diff_task.py ${config} ${exp_dir}

python -u ${exp_dir}/train.py \
  --config=${config}
