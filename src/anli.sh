#!/bin/bash
declare -a models=(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-1.3B)

for model in ${models[@]}
do
    python anli.py --model ${model} -n 20 -b 32 -d 0
done


declare -a models=(distilbert-base-uncased bert-base-uncased bert-large-uncased distilroberta-base roberta-base roberta-large albert-base-v2 albert-large-v2 albert-xlarge-v2 albert-xxlarge-v2 google/electra-base-generator google/electra-small-generator google/electra-large-generator)

for model in ${models[@]}
do
    python anli.py --model ${model} -n 20 -b 32 -d 0 --mlm
done