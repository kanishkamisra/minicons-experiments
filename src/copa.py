import torch

from torch.utils.data import DataLoader
from minicons import scorer

def create_hypothesis(choice):
    choice_split = choice.split(" ")
    if choice_split[0] == "I":
        choice = choice
    else:
        choice = " ".join([choice_split[0].lower()] + choice_split[1:])

    return choice

def create_stimuli(instance):
    premise, choice1, choice2, question, label = instance['premise'], instance['choice1'], instance['choice2'], instance['question'], instance['label']

    domain = {
        'cause': ' because',
        'effect': ' so'
    }

    hypothesis1 = create_hypothesis(choice1)
    hypothesis2 = create_hypothesis(choice2)

    premise = premise[:-1] + domain[question]

    if label == 0:
        return premise, hypothesis1, hypothesis2, domain[question].strip()
    else:
        return premise, hypothesis2, hypothesis1, domain[question].strip()
    # return premise, hypothesis1, hypothesis2, domain[question]

data = [{"premise": "The man turned on the faucet.", "choice1": "The toilet filled with water.", "choice2": "Water flowed from the spout.", "question": "effect", "label": 1, "idx": 0}, {"premise": "The bar closed.", "choice1": "it was crowded.", "choice2": "it was 3 AM.", "question": "cause", "label": 1, "idx": 0}]

lm = scorer.IncrementalLMScorer('gpt2-medium')

stimuli = [create_stimuli(d) for d in data]

print(stimuli)

dl = DataLoader(stimuli, batch_size=2)

for batch in dl:
    premise, hypothesis1, hypothesis2, domain = batch
    premise, hypothesis1, hypothesis2, domain = [list(x) for x in [premise, hypothesis1, hypothesis2, domain]]

    lpcn = torch.tensor(lm.partial_score(premise, hypothesis1, reduction=lambda x: x.mean(0).item()))
    lpwn = torch.tensor(lm.partial_score(premise, hypothesis2, reduction=lambda x: x.mean(0).item()))
    lpcd = torch.tensor(lm.partial_score(domain, hypothesis1, reduction=lambda x: x.mean(0).item()))
    lpwd = torch.tensor(lm.partial_score(domain, hypothesis2, reduction=lambda x: x.mean(0).item()))

    print(lpcn - lpcd, lpwn - lpwd, lpcn - lpcd > lpwn - lpwd)



# # premise, hypothesis1, hypothesis2, domain = create_stimuli({"premise": "The man turned on the faucet.", "choice1": "The toilet filled with water.", "choice2": "Water flowed from the spout.", "question": "effect", "label": 1, "idx": 0})
# # premise, hypothesis1, hypothesis2, domain = create_stimuli({"premise": "The bar closed.", "choice1": "it was crowded.", "choice2": "it was 3 AM.", "question": "cause", "label": 1, "idx": 0})



# lpcn = lm.partial_score([premise], [hypothesis1], reduction=lambda x: x.mean(0).item())[0]
# lpwn = lm.partial_score([premise], [hypothesis2], reduction=lambda x: x.mean(0).item())[0]
# lpcd = lm.partial_score([domain], [hypothesis1], reduction=lambda x: x.mean(0).item())[0]
# lpwd = lm.partial_score([domain], [hypothesis2], reduction=lambda x: x.mean(0).item())[0]

# print(f"Premise: {premise} correct: {hypothesis1} ({lpcn}, {lpcn - lpcd}) wrong: {hypothesis2} ({lpwn}, {lpwn - lpwd})")

# # print(lm.sequence_score(["The bar closed because it was 3AM.", "The bar closed because it was crowded."], reduction = lambda x: x.mean(0).item()))

# # print(lm.token_score(["The bar closed because it was 3AM.", "The bar closed because it was crowded."]))

# print(lm.prime_text([premise], [hypothesis1]))

# lpcn = lm.partial_score([premise], [hypothesis1], reduction=lambda x: x.sum(0).item())[0]
# lpwn = lm.partial_score([premise], [hypothesis2], reduction=lambda x: x.sum(0).item())[0]
# lpcd = lm.partial_score([domain], [hypothesis1], reduction=lambda x: x.sum(0).item())[0]
# lpwd = lm.partial_score([domain], [hypothesis2], reduction=lambda x: x.sum(0).item())[0]

# print(f"Premise: {premise} correct: {hypothesis1} ({lpcn}, {lpcn - lpcd}) wrong: {hypothesis2} ({lpwn}, {lpwn - lpwd})")

# print(lm.partial_score([premise], [hypothesis1], reduction=lambda x: (x.sum(0).item(), x.mean(0).item())))
# print(lm.partial_score([premise], [hypothesis2], reduction=lambda x: (x.sum(0).item(), x.mean(0).item())))
# print(lm.partial_score([domain], [hypothesis1], reduction=lambda x: (x.sum(0).item(), x.mean(0).item())))
# print(lm.partial_score([domain], [hypothesis2], reduction=lambda x: (x.sum(0).item(), x.mean(0).item())))

'''
what we want to do:

get log P()

'''