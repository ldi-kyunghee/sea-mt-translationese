MODEL=$1
DATASET=$2
IS_VL=$3

export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2 uv run accelerate launch --use_deepspeed --config_file configs/accelerate_single_gpu_config.yaml --main_process_port 0 src/train/sft.py \
--model ${MODEL} ${IS_VL} \
--dataset_name_or_path ${DATASET} \
--bf16 \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=16 \
--per_device_eval_batch_size=8 \
--eval_accumulation_step=16 \
--eval_steps=5 \
--warmup_ratio=0.1 \
--evaluation_strategy='steps' \
--num_train_epochs=5 \
--weight_decay=0.1 \
--learning_rate=2e-5 \
--lr_scheduler='cosine' \
--max_seq_length=2048 \
--logging_steps=5 \
--save_steps=5 \
--save_total_limit 2 \
--report_to="wandb" \
--loss_type='nll' \
--lora_r=64 \
--lora_alpha=256 \
--lora_dropout=0.0 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--lora_bias="none"
exit