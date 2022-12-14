#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=9:00:00
#SBATCH --job-name=mae_lin_core50
#SBATCH --output=mae_lin_core50_%A_%a.out
#SBATCH --array=0-11

module purge
module load cuda/11.3.1

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say s a y say s a y say s a y say s a y)
ARCHS=(vit_large_patch16 vit_large_patch16 vit_large_patch16 vit_large_patch16 vit_base_patch16 vit_base_patch16 vit_base_patch16 vit_base_patch16 vit_small_patch16 vit_small_patch16 vit_small_patch16 vit_small_patch16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH

# core50
python -u /scratch/eo41/mae/eval_linear.py \
	--model ${ARCH} \
	--resume "/scratch/eo41/mae/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_${MODEL} \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 8 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/mae/evals/core50" \
	--train_data_path "/vast/eo41/data/core50/train" \
	--val_data_path "/vast/eo41/data/core50/val" \
	--num_labels 50
	
echo "Done"
