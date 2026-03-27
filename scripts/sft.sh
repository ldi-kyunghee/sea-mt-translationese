#!/usr/bin/bash

#SBATCH -J space
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out
#SBATCH -e ./logs/slurm-err-%A.out

MODEL=$1
DATASET=$2
DATA_SIZE=$3
IS_VL=$4

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training/sft.py \
--model ${MODEL} ${IS_VL} \
--dataset ${DATASET} \
--data_size ${DATA_SIZE} \
--bf16 \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=16 \
--per_device_eval_batch_size=8 \
--eval_accumulation_step=16 \
--eval_steps=5 \
--evaluation_strategy='steps' \
--num_train_epochs=10 \
--weight_decay=0.1 \
--learning_rate=2e-5 \
--lr_scheduler='cosine' \
--max_seq_length=2048 \
--logging_steps=5 \
--save_steps=5 \
--report_to="wandb" \
--lora_r=64 \
--lora_alpha=256 \
--lora_dropout=0.0 \
--lora_bias="none"
exit