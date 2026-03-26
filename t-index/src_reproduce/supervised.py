import os
import yaml
import argparse
import pandas as pd
from glob import glob
from types import SimpleNamespace

import torch
from datasets import Dataset
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from openrlhf.models import get_llm_for_sequence_regression
from tqdm import tqdm
from openrlhf.datasets import RewardDataset


from utils import format_messages


def rm(reward_model, tokenizer, data, batch_size, max_length):
    
    prompts = [
        msgs[0]["content"] for msgs in
        data[data.completion_label==1].messages.tolist()
    ]
    chosen = [
        msgs[1]["content"] for msgs in
        data[data.completion_label==1].messages.tolist()
    ]
    rejected = [
        msgs[1]["content"] for msgs in
        data[data.completion_label==0].messages.tolist()
    ]
    data = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    # init a psuedo strategy
    class Strategy:
        def __init__(self):
            self.args = SimpleNamespace(
                prompt_key="prompt",
                chosen_key="chosen",
                rejected_key="rejected",
                apply_chat_template=None,
            )

    dataset = RewardDataset(
        data,
        tokenizer,
        max_length,
        Strategy(),
        input_template=None,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
            
    def concatenated_forward(model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks


    with torch.no_grad():
        rewards = []
        for data in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            chosen_ids, c_mask, reject_ids, r_mask, margin = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_reward, reject_reward, _ = concatenated_forward(
                reward_model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask
            )
            chosen_reward = chosen_reward.flatten().cpu().tolist()
            reject_reward = reject_reward.flatten().cpu().tolist()
            for c, r in zip(chosen_reward, reject_reward):
                rewards.append(c)
                rewards.append(r)

    return rewards


def dpo(reward_model, tokenizer, data, batch_size, max_length):
    
    prompts = [
        msgs[0]["content"] for msgs in
        data[data.completion_label==1].messages.tolist()
    ]
    chosen = [
        msgs[1]["content"] for msgs in
        data[data.completion_label==1].messages.tolist()
    ]
    rejected = [
        msgs[1]["content"] for msgs in
        data[data.completion_label==0].messages.tolist()
    ]
    data = Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    # init a psuedo strategy
    class Strategy:
        def __init__(self):
            self.args = SimpleNamespace(
                prompt_key="prompt",
                chosen_key="chosen",
                rejected_key="rejected",
                apply_chat_template=None,
            )

    dataset = RewardDataset(
        data,
        tokenizer,
        max_length,
        Strategy(),
        input_template=None,
        is_dpo=True,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
            
    def concatenated_forward(model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask, prompt_len):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, response_mask = concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask, prompt_len)
        logits = model(input_ids, attention_mask=att_masks).logits[:, :-1]
        input_ids = input_ids[:, 1:]
        response_mask = response_mask[:, 1:].bool()
        log_lklh = logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~response_mask, 0)
        all_values = log_lklh.sum(dim=1) / response_mask.sum(dim=1)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        return chosen_rewards, rejected_rewards

    def concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask, prompt_len):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # right pad
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        response_masks = att_masks.clone()
        
        prompt_len = prompt_len + prompt_len
        for i in range(att_masks.size(0)):
            response_masks[i, :prompt_len[i]] = 0
        return inputs_ids, att_masks, response_masks


    with torch.no_grad():
        rewards = []
        for data in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            chosen_ids, c_mask, reject_ids, r_mask, prompt_len = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_reward, reject_reward = concatenated_forward(
                reward_model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask, prompt_len
            )
            chosen_reward = chosen_reward.flatten().cpu().tolist()
            reject_reward = reject_reward.flatten().cpu().tolist()
            for c, r in zip(chosen_reward, reject_reward):
                rewards.append(c)
                rewards.append(r)

    return rewards

def xlmr(clf_model, tokenizer, data, batch_size, max_length):
    max_length = min(max_length, 512)
    def preprocess_function(examples):
        sentences = [msgs[1]["content"].split("\n\n")[-1] for msgs in examples["messages"]]
        result = tokenizer(sentences, padding=True, max_length=max_length, truncation=True)
        # result["label"] = examples["completion_label"]
        return result

    data = Dataset.from_pandas(data)

    data = data.map(
        preprocess_function,
        batched=True,
        remove_columns=data.column_names,
        desc="Running tokenizer on dataset",
    )
    data.set_format("torch", columns=["input_ids", "attention_mask"])

    with torch.no_grad():
        scores = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = {k: v.to(clf_model.device) for k, v in data[i:i+batch_size].items()}
            logits = clf_model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            scores.extend(predictions.cpu().tolist())

    return scores


def main(args):
    results = { 
        "file_path": [],
        "messages_for_positive_model": [],
        "messages_for_negative_model": [],
        "label": [],
        # "bt_rm": [],
        # "dpo_aligned": []
    }

    def load_rm(args):
        reward_model = get_llm_for_sequence_regression(
            args.model_path,
            "reward",
            bf16=True,
            init_value_head=False,
        )
        return reward_model


    def load_dpo(args):
        reward_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        return reward_model

    def load_clf(args):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        return model

    if args.model_type == "rm":
        score_key = "bt_rm"
        model = load_rm(args)
        score = rm
    elif args.model_type == "dpo":
        score_key = "dpo_aligned"
        model = load_dpo(args)
        score = dpo
    elif args.model_type == "clf":
        score_key = "clf"
        model = load_clf(args)
        score = xlmr

    results[score_key] = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    for path in glob(args.data_path):

        dataset_for_positive_model, dataset_for_negative_model = format_messages(
            path,
            args.prompt_field,
            args.completion_positive_field,
            args.completion_negative_field,
            args.prompt_template_positive,
            args.prompt_template_negative,
        )

        scores = score(model, tokenizer, dataset_for_positive_model, args.batch_size, args.max_length)
        results["file_path"].extend([path] * len(scores))
        results["messages_for_positive_model"].extend(dataset_for_positive_model.messages.tolist())
        results["messages_for_negative_model"].extend(dataset_for_negative_model.messages.tolist())
        results["label"].extend(dataset_for_positive_model.completion_label.tolist())
        results[score_key].extend(scores)
    
    df = pd.DataFrame(results)

    output_dir = os.path.dirname(args.output_file)
    output_dir = output_dir if output_dir else "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_json(args.output_file, index=False, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_negative", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt_field", type=str, default=None)
    parser.add_argument("--completion_positive_field", default=None)
    parser.add_argument("--completion_negative_field", default=None)
    parser.add_argument("--prompt_template_positive", default=None)
    parser.add_argument("--prompt_template_negative", default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--model_args", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_type", type=str, choices=["rm", "dpo", "clf"])
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config_args = SimpleNamespace(**config)
        # overwrite config args with command line args
        for k, v in vars(args).items():
            if v is not None:
                setattr(config_args, k, v)
        args = config_args
    print(args)

    main(args)
            