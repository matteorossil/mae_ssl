#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=2K_b_lin_probe
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

srun \
	singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
    --overlay /scratch/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_linprobe.py \
												--model vit_small_patch16 --cls_token \
												--finetune /vast/mr6744/pretrained_models/run2/vit_s_16_200K_blur_checkpoint.pth \
												--batch_size 128 \
												--epochs 100 \
												--num_workers 16 \
												--output_dir /vast/mr6744/finetuned_models/ \
												--blr 0.1 \
												--weight_decay 0.0 \
												--nb_classes 1000 \
												--data_path /scratch/work/public/ml-datasets/imagenet/"
