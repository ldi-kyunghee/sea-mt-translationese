from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from tqdm import tqdm
from random import sample
import torch
import argparse
import json
import os

torch.cuda.empty_cache()

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='ms')
    parser.add_argument('--eng_step', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no-few_shot', dest='few_shot', action='store_false')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_file', type=str)

    parser.set_defaults(few_shot=False)
    parser.set_defaults(eng_step=False)
    return parser

def system_prompt(examples, src_lang, tgt_lang):
    return f"""### Translate the given text from {src_lang} to {tgt_lang}. Here are a few examples:
    {src_lang}: {examples[0][args.src_lang]}
    {tgt_lang}: {examples[0][args.tgt_lang]}
    
    {src_lang}: {examples[1][args.src_lang]}
    {tgt_lang}: {examples[1][args.tgt_lang]}
    
    {src_lang}: {examples[2][args.src_lang]}
    {tgt_lang}: {examples[2][args.tgt_lang]}
    
    {src_lang}: {examples[3][args.src_lang]}
    {tgt_lang}: {examples[3][args.tgt_lang]}
    
    {src_lang}: {examples[4][args.src_lang]}
    {tgt_lang}: {examples[4][args.tgt_lang]}"""


parser = init_parser()
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', quantization_config=BitsAndBytesConfig(load_in_4bit=True))
tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if 'flores' in args.dataset.lower():
    if args.src_lang == 'en':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'eng_Latn', split='devtest')
    elif args.src_lang == 'ms':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'zsm_Latn', split='devtest')
    elif args.src_lang == 'th':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'tha_Thai', split='devtest')

    if args.tgt_lang == 'en':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'eng_Latn', split='devtest')
    elif args.tgt_lang == 'ms':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'zsm_Latn', split='devtest')
    elif args.tgt_lang == 'th':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'tha_Thai', split='devtest')


    data = []
    for src, tgt in zip(src_dataset, tgt_dataset):
        data.append({
            'src': src['text'],
            'ref': tgt['text']
        })
    dataset = Dataset.from_list(data)
elif 'wmt24' in args.dataset.lower():
    dataset = load_dataset('google/wmt24pp', 'en-fil_PH', split='train')
    dataset = dataset.rename_columns({'source': 'src', 'target': 'ref'})
else:
    raise ValueError("Invalid dataset name")

prompt = "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: "

if args.few_shot:
    if args.src_lang == 'en':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'eng_Latn', split='dev')
    elif args.src_lang == 'ms':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'zsm_Latn', split='dev')
    elif args.src_lang == 'th':
        src_dataset = load_dataset('openlanguagedata/flores_plus', 'tha_Thai', split='dev')

    if args.tgt_lang == 'en':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'eng_Latn', split='dev')
    elif args.tgt_lang == 'ms':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'zsm_Latn', split='dev')
    elif args.tgt_lang == 'th':
        tgt_dataset = load_dataset('openlanguagedata/flores_plus', 'tha_Thai', split='dev')
    
    sample_set = []
    for src, tgt in zip(src_dataset, tgt_dataset):
        sample_set.append({
            args.src_lang: src['text'],
            args.tgt_lang: tgt['text']
        })

    query = """\n\n### {src_lang}: {src}\n### {tgt_lang}: """
preds = []

lang_map = {'en': 'English', 'ms': 'Malay', 'th': 'Thai'}

for i, item in enumerate(tqdm(dataset, desc='Predicting')):
    if args.eng_step:
        if args.few_shot:
            samples = sample(sample_set, 5)
            prompt = system_prompt(samples, lang_map[args.src_lang], lang_map['en']) + query

        text = prompt.format(src_lang=lang_map[args.src_lang], src=item['src'], tgt_lang=lang_map['en'])
        inputs = tokenizer(text, return_tensors='pt').to('cuda:0')
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = min(2048, int(input_length)+512)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)
        generated_ids = outputs[0][input_length:]
        eng = tokenizer.decode(outputs[0], skip_special_tokens=True).split(f'English:')[-1].strip()
        if '\n' in eng:
            eng = eng.split('\n')[0].strip()

        if args.few_shot:
            prompt = system_prompt(samples, lang_map['en'], lang_map[args.tgt_lang]) + query
        text = prompt.format(src_lang='en', src=eng, tgt_lang=lang_map[args.tgt_lang])
        inputs = tokenizer(text, return_tensors='pt').to('cuda:0')
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = min(2048, int(input_length)+512)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)
        generated_ids = outputs[0][input_length:]
        mt = tokenizer.decode(outputs[0], skip_special_tokens=True).split(f'{lang_map[args.tgt_lang]}:')[-1].strip()

    else:
        if args.few_shot:
            samples = sample(sample_set, 5)
            prompt = system_prompt(samples, lang_map[args.src_lang], lang_map[args.tgt_lang]) + query

        text = prompt.format(src_lang=lang_map[args.src_lang], src=item['src'], tgt_lang=lang_map[args.tgt_lang])
        inputs = tokenizer(text, return_tensors='pt').to('cuda:0')
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = min(2048, int(input_length)+512)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)
        generated_ids = outputs[0][input_length:]
        mt = tokenizer.decode(outputs[0], skip_special_tokens=True).split(f'{lang_map[args.tgt_lang]}:')[-1].strip()
    if 'flores' in args.dataset:
        preds.append({
            'src': item['src'],
            'mt': mt,
            'ref': item['ref']
        })
    else:
        preds.append({
            'src': item['src'],
            'mt': mt,
        })

os.makedirs('evaluation/results', exist_ok=True)
with open(f'evaluation/results/{args.output_file}.json', 'w') as file:
    json.dump(preds, fp=file, indent=2)