#!/bin/bash
#SBATCH --job-name=dene-thesis
#SBATCH --output=slurm.log
#SBATCH --error=slurm.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=120G
#SBATCH --partition=blue

source /home/nhoffman/env/setup.sh
source /home/nhoffman/.venv/bin/activate

cd "$PWD"
uv pip install --reinstall . 
luigid --background --pidfile ./luigi.pid --logdir ./luigi.log &
luigi --module thesis_analysis RunAll --global-parameters-username="nhoffman" --global-parameters-hostname="ernest.phys.cmu.edu" --workers=32
