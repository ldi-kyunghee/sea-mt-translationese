import os
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
from score_functions import (
    featurize_fn,
    score_log_likelihood,
    score_negative_entropy,
    score_fast_detectgpt,
    get_mean_cov_inv,
    get_tv_score_ID_info,
    trajectory_volatility_fn,
    mahalanobis_distance_fn
)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_positive)

    my_format_messages = partial(
        format_messages,
        prompt_field=args.prompt_field,
        completion_positive_field=args.completion_positive_field,
        completion_negative_field=args.completion_negative_field,
        prompt_template_positive=args.prompt_template_positive,
        prompt_template_negative=args.prompt_template_negative
    )        

    def load_dataset(path):
        df_for_model_positive, df_for_model_negative = my_format_messages(path)
        dataset_for_model_positive = datasets.Dataset.from_pandas(df_for_model_positive)
        dataset_for_model_negative = datasets.Dataset.from_pandas(df_for_model_negative)
        return dataset_for_model_positive, dataset_for_model_negative

    model_positive = AutoModelForCausalLM.from_pretrained(
        args.model_positive, 
        **args.model_args
    ).to(args.device)
    
    model_negative = AutoModelForCausalLM.from_pretrained(
        args.model_negative, 
        **args.model_args
    ).to(args.device)

    model_positive.eval()
    model_negative.eval()

    featurize_positive = partial(featurize_fn, model=model_positive, tokenizer=tokenizer)
    featurize_negative = partial(featurize_fn, model=model_negative, tokenizer=tokenizer)

    # Get stats of hidden_states of samples in validation samples
    # mean and cov_inv for embedding-based methods

    val_for_model_positive, val_for_model_negative = my_format_messages(args.validation_path)
    samples_positive_for_model_positive = val_for_model_positive[val_for_model_positive.completion_label==1]
    samples_negative_for_model_positive = val_for_model_positive[val_for_model_positive.completion_label==0]
    samples_positive_for_model_negative = val_for_model_negative[val_for_model_negative.completion_label==1]
    samples_negative_for_model_negative = val_for_model_negative[val_for_model_negative.completion_label==0]
    
    def get_hs_all_sample_all_layer(df, featurize_func):
        hs_all_sample_all_layer = []
        for i in tqdm(range(0, len(df), args.batch_size), desc="Getting stats"):
            batch = df[i:i+args.batch_size]
            features = featurize_func(batch)
            hs_all_sample_all_layer.extend(features["hidden_states"])
        return hs_all_sample_all_layer
    
    hs_samples_positive_for_model_positive = get_hs_all_sample_all_layer(
        samples_positive_for_model_positive, featurize_positive
    )
    hs_samples_negative_for_model_positive = get_hs_all_sample_all_layer(
        samples_negative_for_model_positive, featurize_positive
    )
    hs_samples_positive_for_model_negative = get_hs_all_sample_all_layer(
        samples_positive_for_model_negative, featurize_negative
    )
    hs_samples_negative_for_model_negative = get_hs_all_sample_all_layer(
        samples_negative_for_model_negative, featurize_negative
    )

    val_stats = {
        "model_positive": {
            "positive": get_mean_cov_inv(hs_samples_positive_for_model_positive),
            "negative": get_mean_cov_inv(hs_samples_negative_for_model_positive)
        },
        "model_negative": {
            "positive": get_mean_cov_inv(hs_samples_positive_for_model_negative),
            "negative": get_mean_cov_inv(hs_samples_negative_for_model_negative)
        }
    }

    ID_info_positive = get_tv_score_ID_info(hs_samples_positive_for_model_positive)
    ID_info_negative = get_tv_score_ID_info(hs_samples_negative_for_model_negative)

    results = { 
        "file_path": [],
        "messages_for_positive_model": [],
        "messages_for_negative_model": [],
        "label": [],
        "log_lklh_positive": [],
        "log_lklh_negative": [],
        "negative_entropy_positive": [],
        "negative_entropy_negative": [],
        "fast_detectgpt_positive": [],
        "fast_detectgpt_negative": [],
        "likelihood_ratios": [],
        "mahalanobis_distance_positive": [],
        "mahalanobis_distance_negative": [],
        "relative_mahalanobis_distance_positive": [],
        "relative_mahalanobis_distance_negative": [],
        "trajectory_volatility_positive": [],
        "trajectory_volatility_negative": []
    }

    for path in glob(args.data_path):
        
        dataset_for_positive_model, dataset_for_negative_model = load_dataset(path)

        for i in tqdm(range(0, len(dataset_for_positive_model), args.batch_size), desc="Evaluating"):
            batch_positive = dataset_for_positive_model[i:i+args.batch_size]
            batch_negative = dataset_for_negative_model[i:i+args.batch_size]
            features_positive = featurize_positive(batch_positive)
            features_negative = featurize_negative(batch_negative)

            log_lklh_positive = score_log_likelihood(**features_positive)
            log_lklh_negative = -score_log_likelihood(**features_negative)
            negtive_entropy_positive = score_negative_entropy(**features_positive)
            negtive_entropy_negative = -score_negative_entropy(**features_negative)
            fast_detectgpt_positive = score_fast_detectgpt(**features_positive)
            fast_detectgpt_negative = -score_fast_detectgpt(**features_negative)
            likelihood_ratios = log_lklh_positive + log_lklh_negative

            distance_to_positive_centroid_of_positive_model = mahalanobis_distance_fn(
                val_stats["model_positive"]["positive"][0],
                val_stats["model_positive"]["positive"][1],
                **features_positive
            )
            distance_to_negative_centroid_of_positive_model = mahalanobis_distance_fn(
                val_stats["model_positive"]["negative"][0],
                val_stats["model_positive"]["negative"][1],
                **features_positive
            )
            distance_to_positive_centroid_of_negative_model = mahalanobis_distance_fn(
                val_stats["model_negative"]["positive"][0],
                val_stats["model_negative"]["positive"][1],
                **features_negative
            )
            distance_to_negative_centroid_of_negative_model = mahalanobis_distance_fn(
                val_stats["model_negative"]["negative"][0],
                val_stats["model_negative"]["negative"][1],
                **features_negative
            )
            mahalanobis_distance_positive = -distance_to_positive_centroid_of_positive_model
            mahalanobis_distance_negative = distance_to_negative_centroid_of_negative_model
            relative_mahalanobis_distance_positive = distance_to_negative_centroid_of_positive_model - distance_to_positive_centroid_of_positive_model
            relative_mahalanobis_distance_negative = distance_to_negative_centroid_of_negative_model - distance_to_positive_centroid_of_negative_model
            trajectory_volatility_positive = -trajectory_volatility_fn(
                ID_info_positive, **features_positive
            )
            trajectory_volatility_negative = trajectory_volatility_fn(
                ID_info_negative, **features_negative
            )

            results["file_path"].extend([path] * args.batch_size)
            results["messages_for_positive_model"].extend(batch_positive["messages"])
            results["messages_for_negative_model"].extend(batch_negative["messages"])
            results["label"].extend(batch_positive["completion_label"])
            
            results["log_lklh_positive"].extend(log_lklh_positive.tolist())
            results["log_lklh_negative"].extend(log_lklh_negative.tolist())
            results["negative_entropy_positive"].extend(negtive_entropy_positive.tolist())
            results["negative_entropy_negative"].extend(negtive_entropy_negative.tolist())
            results["fast_detectgpt_positive"].extend(fast_detectgpt_positive.tolist())
            results["fast_detectgpt_negative"].extend(fast_detectgpt_negative.tolist())
            results["likelihood_ratios"].extend(likelihood_ratios.tolist())
            results["mahalanobis_distance_positive"].extend(mahalanobis_distance_positive.tolist())
            results["mahalanobis_distance_negative"].extend(mahalanobis_distance_negative.tolist())
            results["relative_mahalanobis_distance_positive"].extend(relative_mahalanobis_distance_positive.tolist())
            results["relative_mahalanobis_distance_negative"].extend(relative_mahalanobis_distance_negative.tolist())
            results["trajectory_volatility_positive"].extend(trajectory_volatility_positive.tolist())
            results["trajectory_volatility_negative"].extend(trajectory_volatility_negative.tolist())

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
    parser.add_argument("--validation_path", type=str, default=None)
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
            