

training:
	sbatch -p Abaki -q abaki ./slurm/training_abaki.sh "$(STYLE)" "$(VERSION)"

training-nvidia:
	sbatch ./slurm/training.sh "$(STYLE)" "$(VERSION)"

inference:
	sbatch -p Abaki -q abaki ./slurm/inference_abaki.sh "$(STYLE)" "$(VERSION)"

inference-nvidia:
	sbatch ./slurm/training.sh "$(STYLE)" "$(VERSION)"