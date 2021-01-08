#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH -t 4-00:00:00
module add cuda/9.0
module add cudnn/7-cuda-9.0
source ~/keras/bin/activate
python -u racetrack.py
