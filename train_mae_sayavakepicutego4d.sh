#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_mae_sayavakepicutego4d
#SBATCH --output=train_mae_sayavakepicutego4d_%A_%a.out
#SBATCH --array=0-12

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

DATAS=(
	"sayavakepicutego4d_{000000..000017}" 
	"sayavakepicutego4d_0.1_1_{000000..000001}" 
	"sayavakepicutego4d_0.1_2_{000000..000001}" 
	"sayavakepicutego4d_0.1_3_{000000..000001}" 
	"sayavakepicutego4d_0.01_1_000000" 
	"sayavakepicutego4d_0.01_2_000000" 
	"sayavakepicutego4d_0.01_3_000000" 
	"sayavakepicutego4d_0.001_1_000000" 
	"sayavakepicutego4d_0.001_2_000000" 
	"sayavakepicutego4d_0.001_3_000000" 
	"sayavakepicutego4d_0.0001_1_000000" 
	"sayavakepicutego4d_0.0001_2_000000" 
	"sayavakepicutego4d_0.0001_3_000000"
	)

SAVES=(
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

DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}
SAVE=${SAVES[$SLURM_ARRAY_TASK_ID]}

echo $DATA
echo $SAVE

# # vit-s/14
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_small_patch14' \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vits14/${SAVE}_vits14_checkpoint.pth" \
# 	--batch_size_per_gpu 512 \
# 	--num_workers 16 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vits14" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vits14"

# # vit-b/14
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_base_patch14' \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vitb14/${SAVE}_vitb14_checkpoint.pth" \
# 	--batch_size_per_gpu 128 \
# 	--num_workers 16 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vitb14" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vitb14"

# # vit-l/14
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_large_patch14' \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vitl14/${SAVE}_vitl14_checkpoint.pth" \
# 	--batch_size_per_gpu 256 \
# 	--num_workers 16 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vitl14" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vitl14"

# # vit-h/14
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_huge_patch14' \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14/${SAVE}_vith14_checkpoint.pth" \
# 	--batch_size_per_gpu 256 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--min_lr 0.0001 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vith14" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vith14"

# # vit-h/14 @ 448px
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_huge_patch14_448' \
# 	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14_448/${SAVE}_vith14_448_checkpoint.pth" \
# 	--input_size 448 \
# 	--mask_ratio 0.8 \
# 	--batch_size_per_gpu 42 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--min_lr 0.0001 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vith14_448" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vith14_448"

# vit-h/14 @ 476 px
srun python -u /scratch/eo41/mae/train_mae.py \
	--model 'mae_vit_huge_patch14_476' \
 	--resume "" \
	--input_size 476 \
	--mask_ratio 0.8 \
	--batch_size_per_gpu 34 \
	--num_workers 16 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vith14_476" \
	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
	--save_prefix "${SAVE}_vith14_476"

echo "Done"