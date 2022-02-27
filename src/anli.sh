#!/bin/bash
declare -a models=(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt)

for model in ${models[@]}
do
    python anli.py --model ${model} -n 20 -b 32 -d 0
done