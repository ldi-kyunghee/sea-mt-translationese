from transformers import AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, apply_chat_template
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from tokenizers import AddedToken
from malaya.tokenizer import SentenceTokenizer
from nltk.tokenize import sent_tokenize
from evaluate import load
import random
import numpy as np
import wandb
import argparse
import torch
import gc
import os

torch.cuda.empty_cache()
random.seed(42)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model to fine-tuned [qwen | qwen-instruct | llama | llama-instruct | gemma]", default='gemma')
    parser.add_argument('--is_vl', action='store_true')
    parser.add_argument('--dataset', type=str, help='Dataset to be used', default='ntrex-128-SEA')
    parser.add_argument('--data_size', default="full")
    parser.add_argument('--prompt_type', type=str, help='Prompt language')
    parser.add_argument('--quant_type', type=str, help='Quantization method', default='bnb')
    parser.add_argument('--full_finetuning', action='store_true')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--packing', action='store_true')
    parser.add_argument('--no-packing', dest='packing', action='store_false')
    parser.add_argument('--per_device_train_batch_size', type=int, default=5)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--eval_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--lr_scheduler', type=str, default='inverse_sqrt')
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--evaluation_strategy', type=str, default=None)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--save_total_limit', type=int, default=-1)
    parser.add_argument('--use_liger_kernel', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no-bf16', dest='bf16', action='store_false')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=32)

    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_bias', type=str, default='none')

    parser.set_defaults(is_vl=False)
    parser.set_defaults(packing=False)
    parser.set_defaults(bf16=False)
    parser.set_defaults(fp16=False)
    parser.set_defaults(full_finetuning=False)
    parser.set_defaults(bidirectional=False)
    return parser

def format_ALT(examples):
    en_texts = [example for example in examples['text_2']]
    ms_texts = [example for example in examples['text_1']]
    
    data = {"src": [], "ref": [], "src_lang": [], "tgt_lang": []}
    for ms_text, en_text in zip(en_texts, ms_texts):
        ms_sents = ms_tokenizer.tokenize(ms_text, 5)
        en_sents = sent_tokenize(en_text)

        if not len(ms_sents) == len(en_sents):
            ms_sents = sent_tokenize(ms_text)
        
        if len(ms_sents) == len(en_sents):
            for en_sent, ms_sent in zip(en_sents, ms_sents):
                data['src'].append(en_sent)
                data['ref'].append(ms_sent)
                data['src_lang'].append('English')
                data['tgt_lang'].append('Malay')
    return data

def format_func(example):
    prompt = [
        {
            "role": "user", "content": f"Translate the following sentence from {example['src_lang']} to {example['tgt_lang']}.\n\n{example['src_lang']}: {example['src']}\n\n{example['tgt_lang']}: "
        }
    ]

    completion = [
        {
            "role": "assistant", "content": example['ref']
        }
    ]

    return {'prompt': prompt, 'completion': completion}

def format_conversational(examples):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    messages = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}."
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, src_lang, tgt_lang in zip(srcs, refs, src_langs, tgt_langs):
        if 'gemma' in args.model.lower():
            user_prompt = sys_prompt + '\n\n' + user_prompt

        message = [
            {
                "role": "user", "content": [{"type": "text", "text": user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if args.is_vl else user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
            },
            {
                "role": "assistant" if not 'gemma' in args.model.lower() else 'model', "content": [{"type": "text", "text": ref}] if args.is_vl else ref
            }
        ]

        if not 'gemma' in args.model.lower():
            message.append({
                "role": "system", "content": [{"type": "text", "text": sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)}] if args.is_vl else sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)
            })
        messages.append(message)

    return {'messages': messages}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

wandb.login(key=os.environ['WANDB_KEY'])

parser = init_parser()
args = parser.parse_args()

dataset_name = args.dataset.split('/')[-1]

if 'asian_treebank' in dataset_name:
    ms_tokenizer = SentenceTokenizer()

try:
    data_size = int(args.data_size)
except:
    data_size = args.data_size

if not args.full_finetuning:
    experiment_name = f"SFT_{dataset_name}_{data_size}_lr_{args.learning_rate}_ep_{args.num_train_epochs}_wd_{args.weight_decay}_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}"
else: experiment_name = f"SFT_{dataset_name}_{data_size}_lr_{args.learning_rate}_ep_{args.num_train_epochs}_wd_{args.weight_decay}"
wandb.init(
        project=f"SEAMT_{args.model.split('/')[-1]}",
        name=experiment_name,
        config = {
            'epochs': args.num_train_epochs,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'lora': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'bias': args.lora_bias
            }
        },
        reinit=True
    )

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

if args.is_vl:
    from transformers import AutoModelForImageTextToText
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        attn_implementation='sdpa',
        device_map='auto',
        dtype=torch.bfloat16,
    )
else:
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config if not args.full_finetuning else None,
        attn_implementation='eager' if 'gemma' in args.model.lower() else 'sdpa',
        device_map='auto',
        dtype=torch.bfloat16 if not args.full_finetuning else None,
    )

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    #device_map='auto',
    add_eos_token=True,
    padding_side='left'
)

if 'ModelSpace' in args.model:
    gemma_tokenizer = AutoTokenizer.from_pretrained(
        'google/gemma-2-2b-it',
    )

    #tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})

    tokenizer.chat_template = gemma_tokenizer.chat_template
    tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})
    base_model.resize_token_embeddings(len(tokenizer))
    del gemma_tokenizer

def preprocess_dataset(example):
    return apply_chat_template(
        example,
        tokenizer=tokenizer
    )

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

metric = load('sacrebleu', tokenize='spBLEU-1K')
def compute_metrics(pred_eval):
    gc.collect()
    preds, labels = pred_eval
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {'spBLEU': result['score']}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if 'parallel_asian_treebank' in args.dataset:
    dataset = load_dataset(
        args.dataset,
        'parallel_asian_treebank_zlm_eng_seacrowd_t2t',
        trust_remote_code=True
        )
else:
    dataset = load_dataset(args.dataset)

if 'parallel_asian_treebank' in dataset_name:
    dataset = dataset.map(format_ALT, remove_columns=['id', 'text_1', 'text_2', 'text_1_name', 'text_2_name'], batched=True)
    col_names = dataset['train'].column_names
dataset = dataset.map(format_conversational, remove_columns=col_names, batched=True)
dataset = dataset.map(preprocess_dataset)
train_set = dataset['train']
valid_set = dataset['validation' if 'parallel_asian_treebank' in dataset_name else 'valid']


if isinstance(data_size, int):
    train_idx = random.sample(range(len(train_set)), data_size)
    valid_size = round(data_size / 0.7 * 0.3)
    if valid_size < len(valid_set):
        valid_idx = random.sample(range(len(valid_set)), valid_size)
        valid_set = valid_set.select(valid_idx)
    train_set = train_set.select(train_idx)
    

base_model.gradient_checkpointing_enable()

os.makedirs('models', exist_ok=True)
output_dir = f'models/{args.model}_{experiment_name}'

early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.001
    )

data_collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, return_tensors='pt')

training_args = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_accumulation_steps=args.eval_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    eval_strategy=args.evaluation_strategy,
    max_length=args.max_seq_length,
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    completion_only_loss=True,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    packing=args.packing,
    warmup_ratio=0.1,
    bf16=args.bf16,
    fp16=args.fp16,
    max_grad_norm=1.0,
    optim='adamw_8bit',
    lr_scheduler_type=args.lr_scheduler,
    load_best_model_at_end=True,
)

if not args.full_finetuning:
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
    r=args.lora_r, 
    lora_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias=args.lora_bias,
    task_type="CAUSAL_LM",
    use_rslora=True,
    )

trainer = SFTTrainer(
    model=base_model,
    processing_class=tokenizer,
    train_dataset=train_set,
    eval_dataset=valid_set,
    args=training_args,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
    peft_config=peft_config if not args.full_finetuning else None,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)


trainer.train()

trainer.save_model(output_dir)

trainer.model.save_pretrained(f'{output_dir}/final_checkpoint')
tokenizer.save_pretrained(f'{output_dir}/final_checkpoint')

wandb.finish()

if not args.full_finetuning:
    model = PeftModelForCausalLM.from_pretrained(base_model, f'{output_dir}', device_map='auto', torch_dtype=torch.bfloat16)
    del base_model
    model = model.merge_and_unload()

model.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
tokenizer.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)

model.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT_{dataset_name}_{data_size}")
tokenizer.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT_{dataset_name}_{data_size}")
torch.cuda.empty_cache()