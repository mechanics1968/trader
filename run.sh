#!/bin/sh
#SBATCH -p ai-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=8
#SBATCH --time=24:00:00

ARCH=$(uname -m)

. /home/users/hikaru.inoue/.venv_x86_64/bin/activate

python main.py --optimize --n-trials 500
