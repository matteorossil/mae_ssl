#!/bin/bash

##SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --job-name=visualize_mae
#SBATCH --output=visualize_mae_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/mae/visualize_mae.py

echo "Done"
