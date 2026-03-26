import torch
from scipy.spatial.distance import mahalanobis
from tv_score_utils import get_IDinfo, score_trajectory_volatility_fn


@torch.no_grad()
def featurize_fn(examples, model, tokenizer):
    results = { "labels": [], "logits": [], "hidden_states": [] }
    for messages in examples["messages"]:
        input_len = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        outputs = model(input_ids, output_hidden_states=True)
        labels = input_ids[:,input_len:]
        output_feats = [hs[:,input_len:,:].mean(dim=1)[-1] for hs in outputs.hidden_states]
        logits = outputs.logits[:,input_len-1:-1]
        results["labels"].append(labels)
        results["logits"].append(logits)
        results["hidden_states"].append(output_feats)
    return results

def score_log_likelihood(**kwargs):
    scores = []
    for logits, labels in zip(kwargs["logits"], kwargs["labels"]):
        labels = labels.unsqueeze(-1)
        log_likelihood = logits.log_softmax(dim=-1).gather(dim=-1, index=labels).squeeze(-1)
        scores.append(log_likelihood.mean().item())
    return torch.tensor(scores)

def score_negative_entropy(**kwargs):
    scores = []
    for logits in kwargs["logits"]:
        probs = logits.softmax(dim=-1)
        logprobs = torch.log(probs)
        entropy = - (probs * logprobs).sum(dim=-1)
        scores.append(-entropy.mean().item())
    return torch.tensor(scores)

def score_fast_detectgpt(**kwargs):
    """
    Implementation of Fast DetectGPT for translationese measurment.
    The codes are copied from:
    https://github.com/baoguangsheng/fast-detect-gpt
    """
    scores = []
    for logits, labels in zip(kwargs["logits"], kwargs["labels"]):
        logits_ref, logits_score = logits, logits
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        scores.append(discrepancy.item())
    return torch.tensor(scores)

def get_mean_cov_inv(hs_all_sample_all_layer):
    hs_last_layer = torch.stack([
        x[-1].to(torch.float32) for x in hs_all_sample_all_layer
    ])
    mean_mat = hs_last_layer.mean(dim=0)
    cov_inv = torch.diag(1.0 / hs_last_layer.var(dim=0))
    return mean_mat.cpu().numpy(), cov_inv.cpu().numpy()

def mahalanobis_distance_fn(mean, cov_inv, **kwargs):
    scores = []
    for hidden_states in kwargs["hidden_states"]:
        hidden_states = hidden_states[-1].to(torch.float32).cpu().numpy()
        scores.append(mahalanobis(hidden_states, mean, cov_inv))
    return torch.tensor(scores)

def get_tv_score_ID_info(hs_all_sample_all_layer):
    hs_all_sample_all_layer = [
        [x.to(torch.float32).cpu() for x in hs] for hs in hs_all_sample_all_layer
    ]
    return get_IDinfo(hs_all_sample_all_layer)

def trajectory_volatility_fn(ID_info, **kwargs):
    return torch.tensor(score_trajectory_volatility_fn(ID_info, **kwargs))
