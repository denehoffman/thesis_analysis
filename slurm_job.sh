#!/bin/bash
#SBATCH --job-name=slurm-cmd
#SBATCH --output={cwd / 'slurm.log'}
#SBATCH --error={cwd / 'slurm.log'}
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=blue

cd "$PWD" || exit
pip install .
luigid --background --pidfile ./luigi.pid --logdir ./luigi.log &
luigi --module thesis_analysis RunAll \
  --global-parameters-username="nhoffman" \
  --global-parameters-hostname="ernest.phys.cmu.edu" \
  --workers=32
