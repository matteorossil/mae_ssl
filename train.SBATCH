#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:10:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=mae_1M_blur
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	    --overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python -u train_mae_nowds.py \
													--model 'mae_vit_small_patch16' \
													--batch_size_per_gpu 128 \
													--num_workers 16 \
													--lr 0.0003 \
													--min_lr 0.0003 \
													--epochs 200 \
													--output_dir /vast/mr6744/pretrained_models/ \
													--data_path /vast/mr6744/SAYCAM/ \
													--save_prefix 1M_blur"