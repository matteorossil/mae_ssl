#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=mae_eval_linear_wds
#SBATCH --output=mae_eval_linear_wds_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

# for reasons, this should only be run on a single gpu with num_workers=1 for now. I'm sorry.

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say s a y say s a y say s a y)
ARCHS=(vit_large_patch16 vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH

# labeled_s
srun python -u /scratch/eo41/mae/eval_linear_wds.py \
	--model $ARCH \
	--finetune /scratch/eo41/mae/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth \
	--save_prefix ${SUBJECT}_5fps_${MODEL} \
	--batch_size 512 \
	--epochs 500 \
	--num_workers 1 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/mae/evals/labeled_s" \
	--train_data_path "/scratch/eo41/data/labeled_s/labeled_s_train_000000.tar" \
	--val_data_path "/scratch/eo41/data/labeled_s/labeled_s_val_000000.tar" \
	--n_train 2878 \
	--n_val 2878 \
	--nb_classes 26
	
echo "Done"
