from transformers import AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, apply_chat_template
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from tokenizers import AddedToken

from utils import format_ALT, format_conversational, compute_metrics, preprocess_dataset, preprocess_logits_for_metrics
from evaluate import load

from functools import partial
from pathlib import Path
import numpy as np
import random
import math
import wandb
import argparse
import torch
import sys
import gc
import os

torch.cuda.empty_cache()
random.seed(42)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model to fine-tuned",)
    parser.add_argument('--is_vl', action='store_true')
    parser.add_argument('--dataset_name_or_path', type=str, help='Dataset to be used')
    parser.add_argument('--prompt_type', type=str, help='Prompt language')
    parser.add_argument('--quant_type', type=str, help='Quantization method', default='bnb')
    parser.add_argument('--full_finetuning', action='store_true')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--packing', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int, default=5)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--eval_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--lr_scheduler', type=str, default='inverse_sqrt')
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--evaluation_strategy', type=str, default=None)
    parser.add_argument('--completion_only_loss', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--save_total_limit', type=int, default=-1)
    parser.add_argument('--use_liger_kernel', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--loss_type', type=str, default='nll')
    parser.add_argument('--local_rank', type=int, default=32)

    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_bias', type=str, default='none')
    parser.add_argument('--lora_target_modules', nargs='*')

    parser.set_defaults(is_vl=False)
    parser.set_defaults(packing=False)
    parser.set_defaults(bf16=False)
    parser.set_defaults(fp16=False)
    parser.set_defaults(full_finetuning=False)
    parser.set_defaults(bidirectional=False)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    dataset_name = args.dataset_name_or_path.split('/')[-1]

    experiment_name = f"SFT_{dataset_name}_lr_{args.learning_rate}_ep_{args.num_train_epochs}_wd_{args.weight_decay}{'_completion_only_loss' if args.completion_only_loss else ''}_{args.loss_type}_loss_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}"
    
    run = wandb.init(
            project=f"SEAMT_{args.model.split('/')[-1]}",
            name=experiment_name,
            config = {
                'epochs': args.num_train_epochs,
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
                'loss_type': args.loss_type,
                'completion_only_loss': args.completion_only_loss,
                'lora': {
                    'r': args.lora_r,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout,
                    'bias': args.lora_bias
                }
            },
            reinit="finish_previous"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    if args.is_vl:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            #attn_implementation='sdpa',
            device_map='auto',
            dtype=torch.bfloat16,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config if not args.full_finetuning else None,
            #attn_implementation='eager' if 'gemma' in args.model.lower() else 'sdpa',
            device_map='auto',
            dtype=torch.bfloat16 if not args.full_finetuning else None,
        )

    tokenizer = AutoProcessor.from_pretrained(
        args.model,
        add_eos_token=True,
        padding_side='left'
    )

    if Path(args.dataset_name_or_path).exists():
        data_files = {
            split: f"{args.dataset_name_or_path}/{split}.json"
            for split in ['train', 'valid', 'test']
        }
        dataset = load_dataset("json", data_files=data_files)

    else:
        dataset = load_dataset(args.dataset_name_or_path)
        
    formatting_func = partial(format_conversational, tokenizer=tokenizer, is_vl=args.is_vl)
    dataset = dataset.map(formatting_func, batched=True)
    
    #apply_chat_template = partial(preprocess_dataset, tokenizer=tokenizer)
    #dataset = dataset.map(apply_chat_template)
    train_set = dataset['train']
    valid_set = dataset['valid']

    os.makedirs('/data/dania/sea-mt/models', exist_ok=True)
    output_dir = f'/data/dania/sea-mt/models/{args.model}_{experiment_name}'

    compute_metrics = partial(compute_metrics, tokenizer=tokenizer)
    early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        )

    data_collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, return_tensors='pt')

    num_devices = torch.cuda.device_count()
    steps_per_epoch = math.ceil(train_set.num_rows / (args.per_device_train_batch_size * args.gradient_accumulation_steps * num_devices))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=args.evaluation_strategy,
        max_length=args.max_seq_length,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        gradient_checkpointing=True,
        completion_only_loss=args.completion_only_loss,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        packing=args.packing,
        warmup_steps=warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        optim='adamw_8bit',
        lr_scheduler_type=args.lr_scheduler,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='spBLEU',
        loss_type=args.loss_type,
        seed=42,
        eval_on_start=True,
        report_to=args.report_to
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
    r=args.lora_r, 
    lora_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_target_modules,
    bias=args.lora_bias,
    task_type="CAUSAL_LM",
    use_rslora=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=valid_set,
        args=training_args,
        data_collator=data_collator,
        callbacks=[early_stopping_callback] if args.early_stopping else None,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    with run:
        trainer.train()
    run.finish()

    trainer.save_model(output_dir)

    trainer.model.save_pretrained(f'{output_dir}/final_checkpoint')
    tokenizer.save_pretrained(f'{output_dir}/final_checkpoint')

    model = model.merge_and_unload()

    model.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
    tokenizer.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)

    model.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT-SEA")
    tokenizer.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT-SEA")
    torch.cuda.empty_cache()
    gc.collect()