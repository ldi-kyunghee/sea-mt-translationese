#!/usr/bin/bash

#SBATCH -J high-T
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 4-0
#SBATCH -o ../logs/slurm-%A.out
#SBATCH -e ../logs/slurm-err-%A.out

model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5
learning_rate=$6
epoch=$7
batch_size=$8

tar -xvf data/t-index_data.tar.gz -C /local_datasets/
export DATA_DIR="/local_datasets/t-index_data/synthetic"

deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset ${DATA_DIR}/${train_data_dir}/train.jsonl \
   --input_key messages_foreignization \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 8 \
   --max_epochs ${epoch} \
   --pretrain ${model_id} \
   --save_path models/sft/${model}-${train_data_dir}-${max_samples}-${seed}/positive \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 3 \
   --max_samples ${max_samples} \
   --bf16 \
   --use_tensorboard logs/sft/${model}-${train_data_dir}-${max_samples}-${seed}/positive \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --flash_attn \
   --gradient_checkpointing \
   --adam_offload \
   --packing_samples
   
rm -r /local_datasets/t-index_data