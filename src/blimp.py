import csv
import json
import os
import torch

from minicons import scorer
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

'''
What is going on here?
'''

os.environ["TOKENIZERS_PARALLELISM"] = "false"

blimp_results = []

for seed in trange(3):
    # if seed == 0:
    for step_dir in os.listdir(f"../multiberts/seed_{seed}/"):
        # if step_dir == 'step_100000':
        print(f"Working on {seed} - {step_dir}")
        model_path = f'../multiberts/seed_{seed}/{step_dir}'
        if os.path.isdir(model_path):
            step = int(step_dir.split("_")[-1])

            # load minicons model.
            lm = scorer.MaskedLMScorer(model_path, 'cuda:0')

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
                    blimp_results.append([j+1, field, linguistic_term, uid, seed, step, g, b])

with open("../data/results/blimp_multiberts_results_012.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['instance_id', 'field', 'topic', 'phenomena', 'seed', 'step', 'good', 'bad'])
    writer.writerows(blimp_results)