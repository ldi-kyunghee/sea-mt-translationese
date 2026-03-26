#!/usr/bin/bash

#SBATCH -J check
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 4-0
#SBATCH -o ../logs/slurm-%A.out
#SBATCH -e ../logs/slurm-err-%A.out

tar -xvf data/t-index_data.tar.gz -C /local_datasets/

model=$1
model_dir="parallel_asian_treebank_qwen"
python src/unsupervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_positive models/sft/${model}-${model_dir}-6000-10/positive \
    --model_negative models/sft/${model}-${model_dir}-6000-10/negative \
    --output_file results/synthetic_enms_unsupervised_results.jsonl
python src/supervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_path models/rm/${model}-${model_dir}-6000-10 \
    --output_file results/synthetic_enms_supervised_rm_results.jsonl \
    --model_type rm
python src/supervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_path models/dpo/${model}-${model_dir}-6000-10 \
    --output_file results/synthetic_enms_supervised_dpo_results.jsonl \
    --model_type dpo
python src/supervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_path models/clf/xlm-roberta-large-${model_dir}-10 \
    --output_file results/synthetic_enms_supervised_clf_results.jsonl \
    --model_type clf

rm -r /local_datasets/t-index_data
