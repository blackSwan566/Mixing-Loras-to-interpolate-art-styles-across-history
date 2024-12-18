#!/bin/bash

#SBATCH --job-name=diffusion
#SBATCH -p NvidiaAll
#SBATCH --mem=8G
#SBATCH --output=nvidia.out

TASK=$1

source venv/bin/activate

python main.py $TASK
