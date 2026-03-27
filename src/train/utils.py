from transformers import PreTrainedTokenizerBase
from malaya.tokenizer import SentenceTokenizer
from nltk.tokenize import sent_tokenize

from trl import apply_chat_template
from evaluate import load

import numpy as np
import gc


def format_ALT(examples):
    ms_tokenizer = SentenceTokenizer()
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

def format_conversational(examples, model_name, is_vl):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    messages = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}."
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, src_lang, tgt_lang in zip(srcs, refs, src_langs, tgt_langs):
        if 'gemma' in model_name.lower():
            user_prompt = sys_prompt + '\n\n' + user_prompt

        message = [
            {
                "role": "user", "content": [{"type": "text", "text": user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if is_vl else user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
            },
            {
                "role": "assistant" if not 'gemma' in model_name.lower() else 'model', "content": [{"type": "text", "text": ref}] if is_vl else ref
            }
        ]

        if not 'gemma' in model_name.lower():
            message.append({
                "role": "system", "content": [{"type": "text", "text": sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)}] if is_vl else sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)
            })
        messages.append(message)

    return {'messages': messages}

def preprocess_dataset(example, tokenizer: PreTrainedTokenizerBase):
    return apply_chat_template(
        example,
        tokenizer=tokenizer
    )

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

metric = load('sacrebleu', tokenize='spBLEU-1K')

def compute_metrics(pred_eval, tokenizer):
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
