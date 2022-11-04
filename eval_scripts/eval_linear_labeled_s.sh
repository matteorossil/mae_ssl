#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --job-name=mae_lin_labeled_s
#SBATCH --output=mae_lin_labeled_s_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16)
SUBJECTS=(say s a y say s a y say s a y)
ARCHS=(vit_large_patch16 vit_large_patch16 vit_large_patch16 vit_large_patch16 vit_base_patch16 vit_base_patch16 vit_base_patch16 vit_base_patch16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH

# labeled_s
srun python -u /scratch/eo41/mae/eval_linear.py \
	--model $ARCH \
	--resume "/scratch/eo41/mae/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_5fps_${MODEL} \
	--batch_size 1024 \
	--epochs 300 \
	--num_workers 8 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/mae/evals/labeled_s" \
	--train_data_path "/vast/eo41/data/labeled_s" \
	--val_data_path "" \
	--num_labels 26 \
	--split \
	--subsample
	
echo "Done"
