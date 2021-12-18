import argparse
import csv
import json
import os
import torch

from minicons import scorer
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    "-m",
    type=str
)

args = parser.parse_args()

model_name = args.model.replace("/", "_")

lm = scorer.MaskedLMScorer(model_name, 'cuda:0')

blimp_results = []

for file in tqdm(os.listdir("../data/blimp/")):
    stimuli = []
    phenomena = []
    with open(f"../data/blimp/{file}", "r") as f:
        for line in f:
            row = json.loads(line)
            phenomena.append(row)
            stimuli.append([row['sentence_good'], row['sentence_bad']])

    uid = phenomena[0]['UID']
    field = phenomena[0]['field']
    linguistic_term = phenomena[0]["linguistics_term"]

    blimp_dl = DataLoader(stimuli, batch_size=64, num_workers=16)

    good_scores = []
    bad_scores = []
    for i, batch in enumerate(blimp_dl):
        good, bad = batch
        good, bad = list(good), list(bad)
        good_score = lm.sequence_score(good)
        bad_score = lm.sequence_score(bad)

        good_scores.extend(good_score)
        bad_scores.extend(bad_score)

    for j, score in enumerate(zip(good_scores, bad_scores)):
        g, b = score
        blimp_results.append([j+1, field, linguistic_term, uid, g, b])

with open(f"../data/results/blimp_{model_name}_results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['instance_id', 'field', 'topic', 'phenomena', 'good', 'bad'])
    writer.writerows(blimp_results)