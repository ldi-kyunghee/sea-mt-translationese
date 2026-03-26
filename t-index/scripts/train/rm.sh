#!/usr/bin/bash

#SBATCH -J check
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 1-0
#SBATCH -o ../logs/slurm-%A.out
#SBATCH -e ../logs/slurm-err-%A.out

model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5


tar -xvf data/t-index_data.tar.gz -C /local_datasets/
export DATA_DIR="/local_datasets/t-index_data/synthetic"

deepspeed --module openrlhf.cli.train_rm \
   --max_len 1024 \
   --dataset ${DATA_DIR}/${train_data_dir}/train.jsonl \
   --chosen_key messages_foreignization \
   --rejected_key messages_domestication \
   --apply_chat_template \
   --train_batch_size 32 \
   --micro_train_batch_size 8 \
   --max_epochs 3 \
   --pretrain ${model_id} \
   --save_path models/rm/${model}-${train_data_dir}-${max_samples}-${seed} \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 3 \
   --gradient_checkpointing \
   --max_samples ${max_samples} \
   --bf16 \
   --flash_attn \
   --use_tensorboard logs/rm/${model}-${train_data_dir}-${max_samples}-${seed} \
   --learning_rate 4e-6 \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --adam_offload \
   --full_determinism \
   --packing_samples 

rm -r /local_datasets/t-index_data