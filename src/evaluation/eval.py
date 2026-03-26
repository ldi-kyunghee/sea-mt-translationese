from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
from bert_score import score as BERT
from comet import download_model, load_from_checkpoint
from rouge_score.rouge_scorer import RougeScorer
import torch
import argparse
import numpy as np
import json
import gc
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--output_file', type=str)
    return parser


def calc_rouge(data):
    print("Calculating ROUGE")
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1, rouge_l, rouge_2 = [], [], []

    for item in data:
        scores = scorer.score(prediction=item['mt'], target=item['ref'])

        rouge_1.append(scores['rouge1'][2])
        rouge_l.append(scores['rougeL'][2])
        rouge_2.append(scores['rouge2'][2])

    results = {
        'rouge_1': float(np.mean(rouge_1) * 100),
        'rouge_2': float(np.mean(rouge_2) * 100),
        'rouge_l': float(np.mean(rouge_l) * 100),
    }

    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results


def comet_model(data):
    print("Calculating Comet")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    scores = model.predict(data)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')    
    return scores


def xcomet(data):
    print("Calculating XCOMET")
    model_path = download_model('Unbabel/XCOMET-XL')
    model = load_from_checkpoint(model_path)
    scores = model.predict(data)
    del model
    torch.cuda.empty_cache()
    gc.collect()  
    print('Done!')
    return scores


def cometkiwi(data):
    print("Calculating CometKiwi")
    data = list(map(lambda x: {'src': x['src'], 'mt': x['mt']}, data))
    model_path = download_model('Unbabel/wmt22-cometkiwi-da')
    model = load_from_checkpoint(model_path)
    scores = model.predict(data)
    del model
    torch.cuda.empty_cache()
    gc.collect()  
    print('Done!')
    return scores


def calc_bleu(data):
    print("Calculating BLEU")
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []

    smoothie = SmoothingFunction().method4

    for item in data:
        target_text = [item['ref'].split()]
        prediction = item['mt'].split()

        bleu_1.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_2.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_3.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu_4.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

    results = {
        'bleu_1': np.mean(bleu_1) * 100,
        'bleu_2': np.mean(bleu_2) * 100,
        'bleu_3': np.mean(bleu_3) * 100,
        'bleu_4': np.mean(bleu_4) * 100,
        'bleu_avg': (np.mean(bleu_1) + np.mean(bleu_2) + np.mean(bleu_3) + np.mean(bleu_4))/4  * 100
    }
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results


def calc_spbleu(data):
    print("Calculating spBLEU")
    mts = [item['mt'] for item in data]
    refs = [[item['ref']] for item in data]
    metric = BLEU(tokenize='spBLEU-1K')
    scores = metric.corpus_score(mts, refs)
    del metric
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return scores.score


def calc_bert(data):
    print("Calculating BERTScore")
    predictions = [item['mt'] for item in data]
    target = [item['ref'] for item in data]
    P, R, F1 = BERT(cands=predictions, refs=target, lang='ms', device='cuda:0')
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return F1.mean().item() * 100


def calc_meteor(data):
    print("Calculating METEOR")
    meteor = load('meteor')

    scores = []
    for item in data:
        score = meteor.compute(predictions=[item['mt']], references=[item['ref']])
        scores.append(score['meteor'])

    results = np.mean(scores) * 100
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return {'meteor': results}


parser = init_parser()
args = parser.parse_args()

print(torch.cuda.is_available())

with open(f'{args.data_file}', 'r') as file:
    data = json.load(file)


spbleu_score = calc_spbleu(data)
comet_score = comet_model(data)
xcomet_score = xcomet(data)
cometkiwi_score = cometkiwi(data)
bleu_score = calc_bleu(data)
rouge = calc_rouge(data)
bert = calc_bert(data)
meteor_score = calc_meteor(data)


final = {
    'Comet': comet_score.system_score,
    'XCOMET': xcomet_score.system_score,
    'CometKiwi': cometkiwi_score.system_score,
    'BERTScore': bert,
    'spBLEU': spbleu_score,
    'ROUGE': rouge,
    'BLEU': bleu_score,
    'METEOR': meteor_score
}

print(final)

os.makedirs('evaluation/scores', exist_ok=True)
with open(f'evaluation/scores/{args.output_file}', 'w') as file:
    json.dump(final, file, indent=2)