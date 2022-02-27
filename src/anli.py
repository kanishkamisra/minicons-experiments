import os
import json
import csv
import torch
import argparse

from minicons import scorer

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_anli(split = "dev"):
    labels = []
    with open(f"../data/anli/{split}-labels.lst", "r") as f:
        for line in f:
            labels.append(int(line))
            
    anli = []
    with open(f"../data/anli/{split}.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            anli.append([data['obs2'], f"{data['obs1']} {data['hyp1']}", f"{data['obs1']} {data['hyp2']}"])
            
    return anli, labels


parser = argparse.ArgumentParser("Abductive natural language inference")

parser.add_argument(
    "--model",
    "-m",
    default = "gpt2",
    type = str,
    help = "Language Model that will be used to estimate probabilities"
)

parser.add_argument(
    "--nworkers",
    "-n",
    default = 12,
    type = int,
    help = "Number of workers for the dataloader"
)

parser.add_argument(
    "--batchsize",
    "-b",
    default = 32,
    type = int,
    help = "Batch size for inference"
)

parser.add_argument(
    "--device",
    "-d",
    default = -1,
    type = int,
    help = "Device to run inference on (-1 for cpu, else use cuda device)"
)

args = parser.parse_args()

if args.device == -1:
    device = 'cpu'
else:
    device = f'cuda:{args.device}'

lm = scorer.IncrementalLMScorer(args.model, device = device)

splits = ['dev', 'test']

split_results = []

for split in splits:
    anli, labels = get_anli(split)
    anli_dl = DataLoader(anli, num_workers=args.nworkers, batch_size=args.batchsize)

    hyp1_scores = []
    hyp2_scores = []
    for batch in tqdm(anli_dl):
        obs2, hyp1, hyp2 = batch

        hyp1_scores.extend(lm.partial_score(list(hyp1), list(obs2)))
        hyp2_scores.extend(lm.partial_score(list(hyp2), list(obs2)))

    predicted = torch.stack((torch.tensor(hyp1_scores), torch.tensor(hyp2_scores))).argmax(0)+1

    acc = accuracy_score(labels, predicted)
    split_results.append(args.model, split, acc)

with open(f"../results/{args.model}_anli.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'split', 'accuracy'])