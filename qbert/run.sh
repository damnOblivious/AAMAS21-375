#!/bin/bash
#SBATCH -A Research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3072
#SBATCH -t 4-00:00:00
module add cuda/9.0
module add cudnn/7-cuda-9.0
source ~/pytorch-pip/bin/activate
python3 -u examples.py

