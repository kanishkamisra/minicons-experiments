# minicons-experiments
Code and analysis for [minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models](https://arxiv.org/abs/2203.13112)

## Environment setup

Replicate the `minicons` environment using the following code:

```bash
conda env create -r environment.yml

# deactivate the current active environment, if any, and then:

conda activate minicons
```

## Data Details

The paper includes two motivating behavioral analysis of transformer language models that we conduct using `minicons`. Each analysis is based around a particular dataset:

### Benchmark of Linguistic Minimal Pairs (BLiMP)

**Paper:** [BLiMP: The benchmark of linguistic minimal pairs for English](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00321/96452)

**Authors:** Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, Samuel R. Bowman

**Source URL:** [Github](https://github.com/alexwarstadt/blimp/tree/master/data)

**Location in this repo:** `data/blimp` (contains 67 `jsonl` files, each targeting a specific type of linguistic phenomena.)

**Goal of the analysis:** Evaluate LMs by assessing their preference for linguistically acceptable vs. unacceptable sentences differing by a single word.

### Abductive Natural Language Inference

**Paper:** [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739)

**Authors:** Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen-tau Yih, Yejin Choi

**Source URL:** [Project Page](http://abductivecommonsense.xyz/), [Leaderboard](https://leaderboard.allenai.org/anli/submissions/public)

**Location in this repo:** `data/anli` (contains 3 `jsonl` files)

**Goal of the analysis:** Evaluate LMs by assessing their capacity to choose the most plausible explanation given two observations.

## Replication

### Exp 1: Learning Dynamics of Relative Linguistic Acceptability in LMs

**Goal:** Shed light on the capacity of the BERT-base model to make linguistic acceptability judgments as it is pre-trained.

Follow the instructions listed [here](multiberts/README.md) to download and use the MultiBERTs model checkpoints. These are checkpoints of 5 different BERT-base models pre-trained on the same corpus using different random seeds. For each of the 5 runs, the authors also make available checkpoints at various time-steps during the course of pre-training.

After this step, the `multiberts` directory should have 5 new directories with the format `seed{0,1,2,3,4}`, each of which with the following structure:

```
seed{n}
├── step_{m}
│   ├── bert.ckpt.index
│   ├── bert.ckpt.meta
│   ├── checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
```

Next, download the data from [here](https://github.com/alexwarstadt/blimp/tree/master/data) and store it in `data/blimp`.

Run the following command to track and save the learning dynamics of the MultiBERTs:

```bash
python src/blimp.py --device 0 --batchsize 64 --workers 16
```
where `--device` is the cuda device (-1 for CPU), `--batchsize` is ..the batch size., and `--workers` is the number of workers used by the `torch.utils.DataLoader`.

This will create a file called `blimp_multiberts_results.csv` in the `data/results` directory with the following columnar format:

```
instance_id,field,topic,phenomena,seed,step,good,bad
```
where `good` and `bad` stand for log-probabilities for the grammatical and ungrammmatical sentences for each instance in a given blimp phenomenon.

### Exp 2: Unsupervised Abductive Natural Language Inference

TODO

## Citation

If you use `minicons` or the code in this repository, please cite the following paper:

```tex
@article{misra2022minicons,
    title={minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models},
    author={Kanishka Misra},
    journal={arXiv preprint arXiv:2203.13112},
    year={2022}
}
```