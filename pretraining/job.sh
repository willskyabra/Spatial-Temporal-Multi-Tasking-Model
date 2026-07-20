#!/bin/bash
#SBATCH -N 1                    
#SBATCH -n 8                    
#SBATCH --mem=16g                
#SBATCH -J "Time numerical feature mask and recovery"    
#SBATCH -p short
#SBATCH -t 24:00:00                       
#SBATCH --gres=gpu:4           

module load python
module load cuda11.7/toolkit/11.7.1

source

python run.py
