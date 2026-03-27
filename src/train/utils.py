from transformers import PreTrainedTokenizerBase
from evaluate import load

import numpy as np
import gc

def system_prompt_supported(tokenizer):
    try: 
        _ = tokenizer.apply_chat_template([{"role": "system", "content": "Lorem ipsum"}])
        return True
    except:
        return False

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

def format_conversational(examples, tokenizer, is_vl):
    srcs = [example for example in examples['src']]
    refs = [example for example in examples['ref']]
    src_langs = [example for example in examples['src_lang']]
    tgt_langs = [example for example in examples['tgt_lang']] 

    prompts = []
    completions = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}."
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, src_lang, tgt_lang in zip(srcs, refs, src_langs, tgt_langs):
        if not system_prompt_supported(tokenizer):
            user_prompt = sys_prompt + '\n\n' + user_prompt
            prompt = [
                {
                    "role": "user", "content": [{"type": "text", "text": user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if is_vl else user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
                }
            ]
            prompts.append(prompt)

        else:
            prompt = [
                {
                    "role": "system", "content": [{"type": "text", "text": sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)}] if is_vl else sys_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang)
                },
                {
                    "role": "user", "content": [{"type": "text", "text": user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if is_vl else user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
                }
            ]
            

        completion = [
            {
                "role": "assistant" if "assistant" in tokenizer.chat_template else 'model', "content": [{"type": "text", "text": ref}] if is_vl else ref
            }
        ]
        
        prompts.append(prompt)
        completions.append(completion)

    return {'prompt': prompts, 'completion': completions}

def preprocess_dataset(example, tokenizer: PreTrainedTokenizerBase):
    if example.get("prompt") and example.get("completion"):
        return {
            "prompt": tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True),
            "completion": tokenizer.apply_chat_template(example['completion'], tokenize=False)
        }
    return {
        "text": tokenizer.apply_chat_template(
            example,
        )
    }

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

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)