#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem 12G

python run.py
