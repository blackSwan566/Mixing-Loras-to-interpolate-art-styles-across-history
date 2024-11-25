

training:
	sbatch -p Abaki -q abaki ./slurm/training.sh "$(STYLE)" "$(VERSION)"

training-nvidia:
	sbatch ./slurm/training.sh "$(STYLE)" "$(VERSION)"

inference:
	sbatch -p Abaki -q abaki ./slurm/inference.sh "$(STYLE)" "$(VERSION)"

inference-nvidia:
	sbatch ./slurm/training.sh "$(STYLE)" "$(VERSION)"