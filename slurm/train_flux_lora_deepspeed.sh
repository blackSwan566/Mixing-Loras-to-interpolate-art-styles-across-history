#!/bin/bash

#SBATCH --job-name=flux
#SBATCH --nodelist=abakus22
#SBATCH --output=output.out

source venv/bin/activate

# cd x-flux

# accelerate launch train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml"

python train_lora.py
