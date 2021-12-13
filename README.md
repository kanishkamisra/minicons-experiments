# minicons-experiments
Code and analysis for the minicons paper

# Environment setup
TODO: decide on poetry vs conda.

Tentative requirements: pytorch, transformers, minicons.

# Data

The paper includes two motivating behavioral analysis of transformer language models that we conduct using `minicons`.

## Benchmark of Linguistic Minimal Pairs (BLiMP)

Paper: [BLiMP: The benchmark of linguistic minimal pairs for English](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00321/96452)

Authors: Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, Samuel R. Bowman

Source URL for data: [Github](https://github.com/alexwarstadt/blimp/tree/master/data)

Location in this repo: `data/blimp` (contains 67 `jsonl` files, each targeting a specific type of linguistic phenomena.)

Goal of the analysis: Evaluate LMs by assessing their preference for linguistically acceptable vs. unacceptable sentences differing by a single word.

## Choice of Plausible Alternatives

Paper: [Choice of Plausible Alternatives](https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF)

Authors: Melissa Roemmele, Cosmin Adrian Bejan, Andrew S. Gordon

Source URL for data: [Original](https://people.ict.usc.edu/~gordon/copa.html), [SuperGLUE task page](https://super.gluebenchmark.com/tasks)

Location in this repo: `data/COPA` (contains 3 `jsonl` files, each containing a set of premises, two alternatives, and the type of relation between the premise and the correct alternative.)

Goal of the analysis: Evaluate LMs by assessing their capacity to reason about cause-effect relations between sentences.