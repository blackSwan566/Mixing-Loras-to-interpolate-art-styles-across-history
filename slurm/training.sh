#!/bin/bash

#SBATCH --job-name=diffusion
#SBATCH --nodelist=abakus22
#SBATCH --output=output.out

source venv/bin/activate

python train_lora.py
