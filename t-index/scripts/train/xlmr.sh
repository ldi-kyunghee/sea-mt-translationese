#!/usr/bin/bash

#SBATCH -J check
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 1-0
#SBATCH -o ../logs/slurm-%A.out
#SBATCH -e ../logs/slurm-err-%A.out

train_data_dir=$1
seed=$2

tar -xvf data/t-index_data.tar.gz -C /local_datasets/
export DATA_DIR="/local_datasets/t-index_data/synthetic"

accelerate launch --config_file accelerate_config.yaml src/run_classification.py \
    --model_name_or_path FacebookAI/xlm-roberta-large \
    --max_seq_length 512 \
    --shuffle_seed ${seed} \
    --train_file ${DATA_DIR}/${train_data_dir}/train.jsonl \
    --validation_file ${DATA_DIR}/${train_data_dir}/valid.jsonl \
    --output_dir models/clf/xlm-roberta-large-${train_data_dir}-${seed} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_dir logs/clf/xlm-roberta-large-${train_data_dir}-${seed} \
    --logging_steps 1 \
    --save_strategy no \
    --seed ${seed} \
    --bf16 \
    --eval_steps 50 \
    --report_to tensorboard

rm -r /local_datasets/t-index_data