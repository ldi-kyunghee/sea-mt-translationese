import os
import sys
import argparse
import tqdm
from pathlib import Path
from functools import partial

from typing_extensions import Literal

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig

import torch

class TranslationeseIndex:
    """
    Wrapper for Translationese-Index by Liu et al., (2025).
    Original TranslationeseIndex code can be found here: https://github.com/yikang0131/TranslationeseIndex

    Args:
        {to do}
    """
    def __init__(self, positive_model: str | PreTrainedModel, negative_model: str | PreTrainedModel, tokenizer: str | PreTrainedTokenizerBase | None = None):
        if tokenizer is None and isinstance(positive_model, str):
            tokenizer = AutoTokenizer.from_pretrained(positive_model)
        if isinstance(positive_model, str):
            positive_model = AutoModelForCausalLM.from_pretrained(positive_model, dtype=torch.bfloat16)
        if isinstance(negative_model, str):
            negative_model = AutoModelForCausalLM.from_pretrained(negative_model, dtype=torch.bfloat16)
        
        if tokenizer is None:
            tokenizer = positive_model.name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.positive_model = positive_model
        self.negative_model = negative_model
        self.tokenizer = tokenizer

    def format_func(self, examples, lang, labels=[], pairwise=False):
        prompt_pmt1ix = f"Translate the following text to {lang}.\n\n"
        srcs = [example for example in examples['src']]
        mts = [example for example in examples['mt']]

        mt_data = []
        for src, mt in zip(srcs, mts):
            messages = [
                {"role": "user", "content": prompt_pmt1ix + src},
                {"role": "assistant", "content": mt}
            ]
            mt_data.append(messages)
        
        if labels:
            return {"messages": mt_data, "labels": labels}
        
        if pairwise:
            mt1_data = []
            assert examples.get('mt1'), "Dataset must contain a `mt1` column for pairwise evaluation."
            mt1 = [example for example in examples['mt1']]
            for src, mt1 in zip(srcs, mt1):
                messages = [
                    {"role": "user", "content": prompt_pmt1ix + src},
                    {"role": "assistant", "content": mt1}
                ]
                mt1_data.append(messages)
            
            return {"mt1_messages": mt1_data, "mt2_messages": mt_data}

        return {"messages": messages}
    
    def tokenize_func(self, examples, tokenizer, max_length):
        user_input_len = [
            len(item) for item in tokenizer.apply_chat_template(
                [m[:1] for m in examples['messages']],
                add_generation_prompt=True
            )
        ]

        concatinated_input_len = [
            len(item) for item in tokenizer.apply_chat_template(
                examples['messages']
            )
        ]

        input_ids = tokenizer.apply_chat_template(
            examples['messages'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        mask = torch.ones(input_ids.shape, type=torch.bool)
        for i in range(len(examples['messages'])):
            user_len = min(user_input_len[i], max_length)
            concat_len = min(concatinated_input_len[i], max_length)

            mask[i, :user_len] = False
            mask[i, concat_len:] = False

            assert mask[i].sum().item() == concat_len - user_len

        return {"input_ids": input_ids, "score_mask": mask}
    
    @torch.no_grad()
    def get_log_lklh(self, model, input_ids, score_mask):
        logits: torch.Tensor = model(input_ids).logits[:, :-1]
        input_ids = input_ids[:, 1:]
        score_mask: torch.Tensor = score_mask[:, 1:]
        log_lklh = logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).unsqueeze(-1)
        log_lklh = log_lklh.masked_fill(~score_mask, 0)
        mean_log_lklh = log_lklh.sum(dim=1) / score_mask.sum(dim=1)
        return mean_log_lklh
    
    def compute_pairwise(self, data: Dataset | dict | list, lang: str, max_length: int = 1024, batch_size: int = 32, num_proc: int | None = None, device: str = "cuda"):
        positive_model = self.positive_model
        negative_model = self.negative_model

        positive_model.eval()
        negative_model.eval()

        if isinstance(data, dict):
            data = Dataset.from_dict(data)
        elif isinstance(data, list):
            data = Dataset.from_list(data)

        format_fn = partial(self.format_func, lang=lang)

        data = data.map(format_fn, batched=True)

        tokenize_fn = partial(self.tokenize_func, tokenizer=self.tokenizer, max_length=max_length)
        tokenized_data = data.map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=False
        )

        llrs = []

        tokenized_data.set_format(type="torch", columns=["input_ids", "score_mask"], output_all_columns=True)
        for i in tqdm(range(0, len(tokenized_data), batch_size), desc="Evaluating..."):
            batch = tokenized_data[i:i+batch_size]
            inputs = {k: batch[k].to(device) for k in ["input_ids", "score_mask"]}
            log_lklh_positive = self.get_log_lklh(model=positive_model, **inputs)
            log_lklh_negative = self.get_log_lklh(model=negative_model, **inputs)

            llr = log_lklh_positive - log_lklh_negative
            llrs.extend(llr.tolist())

        assert len(llrs) == data.num_rows

        mean_llr = torch.tensor(llrs).sum(dim=-1) / len(llrs)
        mean_score = mean_llr.sigmoid()

        return {"translationese_score": mean_score.item(), "naturalness_score": 1 - mean_score.item()}

    def compute_rewards(self, lklh_natural_refs, lklh_translationese_refs, lklh_natural_preds, lklh_translationese_preds):
        ref_probs = (lklh_natural_refs - lklh_translationese_refs).sigmoid()
        pred_probs = (lklh_natural_preds - lklh_translationese_preds).sigmoid()

        return (ref_probs - pred_probs).item()

    def compute_penalties(self, llr_refs, llr_preds):
        ref_probs = llr_refs.sigmoid()
        pred_probs = llr_preds.sigmoid()

        return (pred_probs - ref_probs).item()

    def compute_pairwise(self, data: Dataset | dict | list, lang: str, max_length: int = 1024, batch_size: int = 32, num_proc: int | None = None, device: str = "cuda", return_rewards=False, return_penalties=False):
        translationese_model = self.positive_model
        natural_model = self.negative_model

        translationese_model.eval()
        natural_model.eval()

        format_fn = partial(self.format_func, lang=lang, pairwise=True)
        data = data.map(
            format_fn,
            batched=True
        )

        tokenize_fn = partial(self.tokenize_func, tokenizer=self.tokenizer, max_length=max_length)

        tokenized_mt1 = data.select_columns("mt1_messages").map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=False
        )

        tokenized_mt2 = data.select_columns("mt2_messages").map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=False
        )

        tokenized_mt1.set_format(type='torch', columns=['input_ids', 'score_mask'], output_all_columns=True)
        tokenized_mt2.set_format(type='torch', columns=['input_ids', 'score_mask'], output_all_columns=True)

        results = {
                "pred_label": [],
            }

        for i in range(0, len(data), batch_size):
            mt1_batch = tokenized_mt1[i:i+batch_size]
            mt2_batch = tokenized_mt2[i:i+batch_size]

            mt1_inputs = {k: mt1_batch[k].to(device) for k in ['input_ids', 'score_mask']}
            mt2_inputs = {k: mt2_batch[k].to(device) for k in ['input_ids', 'score_mask']}

            lklh_natural_mt1 = self.get_log_lklh(natural_model, **mt1_inputs)
            lklh_translationese_mt1 = self.get_log_lklh(translationese_model, **mt1_inputs)

            lklh_natural_mt2s = self.get_log_lklh(natural_model, **mt2_inputs)
            lklh_translationese_mt2s = self.get_log_lklh(translationese_model, **mt2_inputs)

            llr_mt1 = lklh_translationese_mt1 - lklh_natural_mt1
            llr_mt2 = lklh_translationese_mt2s - lklh_natural_mt2s

            reward_func_inputs = {
                "lklh_natural_refs": lklh_natural_mt1,
                "lklh_translationese_refs": lklh_translationese_mt1,
                "lklh_natural_preds": lklh_natural_mt2s,
                "lklh_translationese_preds": lklh_translationese_mt2s
            }

            if return_rewards:
                rewards = self.compute_rewards(**reward_func_inputs)
            if return_penalties:
                penalties = self.compute_penalties(llr_refs=llr_mt1, llr_preds=llr_mt2)

            for mt1, mt2 in zip(llr_mt1, llr_mt2):
                if mt1 > mt2:
                    results['preds'].append("mt1")
                else:
                    results['preds'].append("mt2")

            mt1_translationese_score = torch.tensor(llr_mt1).sum(dim=-1) / len(llr_mt1)
            mt2_translationese_score = torch.tensor(llr_mt2).sum(dim=-1) / len(llr_mt2)

            results['mt1_translationese_score'] = mt1_translationese_score.sigmoid().item()
            results['mt1_natural_score'] = 1 - mt1_translationese_score.sigmoid().item()

            results['mt2_translationese_score'] = mt2_translationese_score.sigmoid().item()
            results['mt2_natural_score'] = 1 - mt2_translationese_score.sigmoid().item()

            if return_rewards:
                results['rewards'] = rewards
            if return_penalties:
                results['penalties'] = penalties

        return results

    def __call__(self, data: Dataset | dict | list, lang: str, max_length: int = 1024, batch_size: int = 32, num_proc: int | None = None, device: str = "cuda", mode: Literal["pointwise", "pairwise"] = "pointwise", return_rewards=False, return_penalties=False):
        if mode == 'pointwise':
            return self.compute_pointwise(
                data=data,
                lang=lang,
                max_length=max_length,
                batch_size=batch_size,
                num_proc=num_proc,
                device=device
            )
        
        elif mode == 'pairwise':
            return self.compute_pairwise(
                data=data,
                lang=lang,
                max_length=max_length,
                batch_size=batch_size,
                num_proc=num_proc,
                device=device,
                return_rewards=return_rewards,
                return_penalties=return_penalties
            )