#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=16:00:00
#SBATCH --job-name=mae_finetune_imagenet
#SBATCH --output=mae_finetune_imagenet_%A_%a.out
#SBATCH --array=0

SUBJECTS=(
	"sayavakepicutego4d" 
	"sayavakepicutego4d_0.1_1" 
	"sayavakepicutego4d_0.1_2" 
	"sayavakepicutego4d_0.1_3" 
	"sayavakepicutego4d_0.01_1" 
	"sayavakepicutego4d_0.01_2" 
	"sayavakepicutego4d_0.01_3" 
	"sayavakepicutego4d_0.001_1" 
	"sayavakepicutego4d_0.001_2" 
	"sayavakepicutego4d_0.001_3" 
	"sayavakepicutego4d_0.0001_1" 
	"sayavakepicutego4d_0.0001_2" 
	"sayavakepicutego4d_0.0001_3"
	)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo $SUBJECT

# vith14 @ 490px
python -u /scratch/eo41/mae/eval_finetune.py \
	--model vit_huge_patch14_490 \
	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14_x/${SUBJECT}_vith14_x_checkpoint.pth" \
	--save_prefix ${SUBJECT}_mae_vith14_490 \
	--input_size 490 \
	--batch_size 10 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.00004 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.02 \
	--num_labels 1000

# # vith14 @ 448px
# python -u /scratch/eo41/mae/eval_finetune.py \
# 	--model vit_huge_patch14_448 \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14_448/${SUBJECT}_vith14_448_checkpoint.pth" \
# 	--save_prefix ${SUBJECT}_mae_vith14_448 \
# 	--input_size 448 \
# 	--batch_size 14 \
# 	--epochs 50 \
# 	--num_workers 16 \
# 	--lr 0.00005 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
# 	--train_data_path "/scratch/work/public/imagenet/train" \
# 	--val_data_path "/scratch/eo41/imagenet/val" \
# 	--frac_retained 0.02 \
# 	--num_labels 1000

# # vith14
# python -u /scratch/eo41/mae/eval_finetune.py \
# 	--model vit_huge_patch14 \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14/${SUBJECT}_vith14_checkpoint.pth" \
# 	--save_prefix ${SUBJECT}_mae_vith14 \
# 	--batch_size 120 \
# 	--epochs 50 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
# 	--train_data_path "/scratch/work/public/imagenet/train" \
# 	--val_data_path "/scratch/eo41/imagenet/val" \
# 	--frac_retained 0.02 \
# 	--num_labels 1000

# # vitl14
# python -u /scratch/eo41/mae/eval_finetune.py \
# 	--model vit_large_patch14 \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vitl14/${SUBJECT}_vitl14_checkpoint.pth" \
# 	--save_prefix ${SUBJECT}_mae_vitl14 \
# 	--batch_size 128 \
# 	--epochs 50 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
# 	--train_data_path "/scratch/work/public/imagenet/train" \
# 	--val_data_path "/scratch/eo41/imagenet/val" \
# 	--frac_retained 0.010147 \
# 	--num_labels 1000

# # vitb14
# python -u /scratch/eo41/mae/eval_finetune.py \
# 	--model vit_base_patch14 \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vitb14/${SUBJECT}_vitb14_checkpoint.pth" \
# 	--save_prefix ${SUBJECT}_mae_vitb14 \
# 	--batch_size 128 \
# 	--epochs 50 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
# 	--train_data_path "/scratch/work/public/imagenet/train" \
# 	--val_data_path "/scratch/eo41/imagenet/val" \
# 	--frac_retained 0.010147 \
# 	--num_labels 1000

# # vits14
# python -u /scratch/eo41/mae/eval_finetune.py \
# 	--model vit_small_patch14 \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vits14/${SUBJECT}_vits14_checkpoint.pth" \
# 	--save_prefix ${SUBJECT}_mae_vits14 \
# 	--batch_size 128 \
# 	--epochs 50 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
# 	--train_data_path "/scratch/work/public/imagenet/train" \
# 	--val_data_path "/scratch/eo41/imagenet/val" \
# 	--frac_retained 0.010147 \
# 	--num_labels 1000

echo "Done"
