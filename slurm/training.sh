#!/bin/bash

#SBATCH --job-name=diffusion
#SBATCH -p NvidiaAll
#SBATCH --mem=8G
#SBATCH --output=output.out

STYLE=$1
VERSION=$2

source venv/bin/activate

python train_lora.py --style $STYLE --version $VERSION
