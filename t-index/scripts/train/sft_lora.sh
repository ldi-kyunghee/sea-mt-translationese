#!/usr/bin/bash

#SBATCH -J check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g7
#SBATCH -t 1-0
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

rm -r ~/.cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDA_ARCH_LIST="80;86;90" 

export CUDA_VISIBLE_DEVICES=0
deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/enms/${train_data_dir}/train.jsonl \
   --input_key messages_foreignization \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size ${batch_size} \
   --max_epochs ${epoch} \
   --pretrain ${model_id} \
   --save_path models/sft/${model}-${train_data_dir}-${max_samples}-${seed}/positive \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 3 \
   --max_samples ${max_samples} \
   --bf16 \
   --gradient_checkpointing \
   --use_liger_kernel \
   --use_tensorboard logs/sft/${model}-${train_data_dir}-${max_samples}-${seed}/positive \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --load_in_4bit \
   --lora_rank 8 \
   --lora_alpha 16 \
   --overlap_comm

deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/enms/${train_data_dir}/train.jsonl \
   --input_key messages_domestication \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size ${batch_size} \
   --max_epochs ${epoch} \
   --pretrain ${model_id} \
   --save_path models/sft/${model}-${train_data_dir}-${max_samples}-${seed}/negative \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 3 \
   --gradient_checkpointing \
   --use_liger_kernel \
   --max_samples ${max_samples} \
   --bf16 \
   --use_tensorboard logs/sft/${model}-${train_data_dir}-${max_samples}-${seed}/negative \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --load_in_4bit \
   --lora_rank 8 \
   --lora_alpha 16 \
   --overlap_comm
   