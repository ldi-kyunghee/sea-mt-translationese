#!/usr/bin/bash

#SBATCH -J check
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v6
#SBATCH -t 4-0
#SBATCH -o ../logs/slurm-%A.out
#SBATCH -e ../logs/slurm-err-%A.out

model=$1
model_dir="parallel_asian_treebank_qwen"
tar -xvf data/t-index_data.tar.gz -C /local_datasets/
python src/t_index.py \
    --config recipes/wild.yaml \
    --data_path /local_datasets/t-index_data/wild/enms/pointwise.jsonl \
    --model_positive models/sft/${model}-${model_dir}-6000-10/positive \
    --model_negative models/sft/${model}-${model_dir}-6000-10/negative \
    --prompt_field source \
    --completion_field translation \
    --output_file results/wild_t_index_results.jsonl
python src/supervised.py \
    --config recipes/wild.yaml \
    --model_path models/rm/${model}-${model_dir}-6000-10 \
    --output_file results/wild_rm_results.jsonl \
    --model_type rm
python src/supervised.py \
    --config recipes/wild.yaml \
    --model_path models/dpo/${model}-${model_dir}-6000-10 \
    --output_file results/wild_dpo_results.jsonl \
    --model_type dpo

rm -r /local_datasets/t-index_data