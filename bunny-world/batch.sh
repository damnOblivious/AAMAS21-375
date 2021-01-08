#!/bin/bash
#SBATCH -A Research
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3048
#SBATCH -t 4-00:00:00
module add cuda/9.0
module add cudnn/7-cuda-9.0
source ~/keras/bin/activate
#python -u pacman.py -p HierarchicalDDPGAgent -x 2000 -n 2050 -l smallInitialGrid -g DirectionalGhost -q
#python -u pacman.py -p DDPGAgent -x 2000 -n 2050 -l smallInitialGrid -g DirectionalGhost -q
python -u pacman.py -p GmQ_Pre -x 2000 -n 2001  -l smallInitialGrid -g DirectionalGhost -q 
#python -u pacman.py -p DQNBaselineAgent -x 5000 -n 5001 -l smallInitialGrid -q
#python -u pacman.py -p GmQAgent -g DirectionalGhost -x 1020 -n 1050 -l initialGrid -q

