import os
import sys
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from glob import glob
from types import SimpleNamespace
from functools import partial

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM


from utils import format_messages


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_positive)

    def load_dataset(path):
        try:
            df = pd.read_json(path, lines=True)
        except:
            df = pd.read_csv(path)
        # df = df.dropna()
        messages = {
            "prompt_template_label": [],
            "messages": []
        }

        for i, row in df.iterrows():
            for prompt_template_label, prompt_template in enumerate([args.prompt_template_negative, args.prompt_template_positive]):
                messages["messages"].append(
                    [
                        {"role": "user", "content": prompt_template.format(input=row[args.prompt_field])},
                        {"role": "assistant", "content": row[args.completion_field]},
                    ]
                )
                messages["prompt_template_label"].append(prompt_template_label)


        df = pd.DataFrame(messages)

        df_for_model_positive = df[df["prompt_template_label"] == 1]
        df_for_model_negative = df[df["prompt_template_label"] == 0]
        dataset_for_model_positive = datasets.Dataset.from_pandas(df_for_model_positive)
        dataset_for_model_negative = datasets.Dataset.from_pandas(df_for_model_negative)
        return dataset_for_model_positive, dataset_for_model_negative

    def tokenize_func(examples):
        # only score assistant output, everything else is masked
        user_input_len = [
            len(i) for i in tokenizer.apply_chat_template(
                [m[:1] for m in examples["messages"]], 
                add_generation_prompt=True
            )
        ]

        concated_input_len = [
            len(i) for i in tokenizer.apply_chat_template(
                examples["messages"]
            )
        ]

        input_ids = tokenizer.apply_chat_template(
            examples["messages"], 
            padding="max_length", 
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )

        mask = torch.ones(input_ids.shape, dtype=torch.bool)

        for i in range(len(examples["messages"])):
            # Adjust lengths to account for truncation
            user_len = min(user_input_len[i], args.max_length)
            concat_len = min(concated_input_len[i], args.max_length)

            # Ensure indices are within bounds
            mask[i, :user_len] = False
            mask[i, concat_len:] = False

            # Debugging information
            # print(f"Sample {i}:")
            # print("Adjusted Mask:", mask[i])
            # print("Mask Sum:", mask[i].sum().item())
            # print("Expected Value:", concat_len - user_len)

            # Assert the mask sum matches the expected value
            assert mask[i].sum().item() == concat_len - user_len

        return { "input_ids": input_ids, "score_mask": mask }
    
    @torch.no_grad()
    def get_log_lklh(model, input_ids, score_mask, output_average):
        # input_ids, score_mask: (batch_size, seq_len)
        logits = model(input_ids).logits[:,:-1]
        input_ids = input_ids[:,1:]
        score_mask = score_mask[:,1:]
        log_lklh = logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~score_mask, 0)
        if output_average:
            return log_lklh.sum(dim=1) / score_mask.sum(dim=1)
        return log_lklh
        
    print("Loading models...")
    if not args.model_args:
        args.model_args = {}
    model_positive = AutoModelForCausalLM.from_pretrained(args.model_positive, **args.model_args).to(args.device)
    model_negative = AutoModelForCausalLM.from_pretrained(args.model_negative, **args.model_args).to(args.device)
    model_positive.eval()
    model_negative.eval()

    print("Models loaded.")
    results = { 
        "file_path": [],
        "messages_for_positive_model": [],
        "messages_for_negative_model": [],
        "log_lklh_positive": [],
        "log_lklh_negative": [],
        "llr": []
    }
    print("Processing data...")
    # print(f"Data path: {args.data_path}")
    for path in glob(args.data_path):
        print(f"Processing {path}...")
        dataset_for_positive_model, dataset_for_negative_model = load_dataset(path)
        dataset_for_positive_model = dataset_for_positive_model.map(
            tokenize_func, 
            batched=True, 
            batch_size=args.batch_size, 
            num_proc=args.num_proc,
            load_from_cache_file=False,
        )
        dataset_for_negative_model = dataset_for_negative_model.map(
            tokenize_func, 
            batched=True, 
            batch_size=args.batch_size, 
            num_proc=args.num_proc,
            load_from_cache_file=False,
        )
        # print(dataset_for_negative_model)
        dataset_for_positive_model.set_format(type="torch", columns=["input_ids", "score_mask"], output_all_columns=True)
        dataset_for_negative_model.set_format(type="torch", columns=["input_ids", "score_mask"], output_all_columns=True)
        
        for i in tqdm(range(0, len(dataset_for_positive_model), args.batch_size), desc="Evaluating"):
            batch_for_positive_model = dataset_for_positive_model[i:i+args.batch_size]
            batch_for_negative_model = dataset_for_negative_model[i:i+args.batch_size]
            inputs_for_positive_model = {k: batch_for_positive_model[k].to(args.device) for k in ["input_ids", "score_mask"]}
            inputs_for_negative_model = {k: batch_for_negative_model[k].to(args.device) for k in ["input_ids", "score_mask"]}
            log_lklh_positive = get_log_lklh(model_positive, output_average=True, **inputs_for_positive_model)
            log_lklh_negative = get_log_lklh(model_negative, output_average=True, **inputs_for_negative_model)
            llr = log_lklh_positive - log_lklh_negative
            results["file_path"].extend([path] * len(batch_for_positive_model["messages"]))
            results["messages_for_positive_model"].extend(batch_for_positive_model["messages"])
            results["messages_for_negative_model"].extend(batch_for_negative_model["messages"])
            results["log_lklh_positive"].extend(log_lklh_positive.tolist())
            results["log_lklh_negative"].extend(log_lklh_negative.tolist())
            results["llr"].extend(llr.tolist())
        
        # check length of each list
        try:
            assert all(len(v) == len(results["file_path"]) for v in results.values())
        except AssertionError:
            # print length of each list
            print({k: len(v) for k, v in results.items()})
            sys.exit(1)

    df = pd.DataFrame(results)

    output_dir = os.path.dirname(args.output_file)
    output_dir = output_dir if output_dir else "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_json(args.output_file, index=False, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_positive", type=str, default=None)
    parser.add_argument("--model_negative", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt_field", type=str, default=None)
    parser.add_argument("--completion_field", default=None)
    parser.add_argument("--prompt_template_positive", default=None)
    parser.add_argument("--prompt_template_negative", default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--model_args", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
            