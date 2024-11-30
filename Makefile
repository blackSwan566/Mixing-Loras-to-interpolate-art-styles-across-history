run-main:
    sbatch ./slurm/main.sh "$(TASK)"

run-main-abaki:
    sbatch -p Abaki -q abaki ./slurm/main_abaki.sh "$(TASK)
