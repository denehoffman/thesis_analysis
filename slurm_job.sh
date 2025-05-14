#!/bin/bash
#SBATCH --job-name=dene-thesis
#SBATCH --output=slurm.log
#SBATCH --error=slurm.log
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --partition=blue

source /home/nhoffman/.venv/bin/activate
source /raid3/nhoffman/root/root_install_313/bin/thisroot.sh

cd "$PWD"
uv pip install --reinstall .

# # Reverse tunnel Luigi dashboard to your local machine
# if [[ -n "$LOCAL_HOSTNAME" && -n "$LOCAL_USERNAME" && -n "$LOCAL_SSH_PORT" ]]; then
#   echo "Setting up reverse tunnel to $LOCAL_USERNAME@$LOCAL_HOSTNAME"
#   ssh -p "$LOCAL_SSH_PORT" -N -R 8082:localhost:8082 "$LOCAL_USERNAME@$LOCAL_HOSTNAME" &
# else
#   echo "Skipping reverse tunnel setup"
# fi

python -c "import laddu; print(f'Cores available: {laddu.available_parallelism()}')"
luigid --background --pidfile ./luigi.pid --logdir ./luigi.log &
GLUEX_USERNAME=nhoffman GLUEX_HOSTNAME=ernest.phys.cmu.edu NUM_THREADS=32 luigi --module thesis_analysis RunAll --workers=32
echo "Done!"
