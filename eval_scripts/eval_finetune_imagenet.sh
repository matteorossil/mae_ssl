#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=4:00:00
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

# imagenet
python -u /scratch/eo41/mae/eval_finetune.py \
	--model vit_large_patch14 \
	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vitl14/${SUBJECT}_vitl14_checkpoint.pth" \
	--save_prefix ${SUBJECT}_mae_vitl14 \
	--batch_size 128 \
	--epochs 25 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.010147 \
	--num_labels 1000
	
echo "Done"
