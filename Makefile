training:
	sbatch -p Abaki -q abaki ./slurm/training.sh

inference:
	sbatch -p Abaki -q abaki ./slurm/inference.sh