#!/bin/bash
declare -a models=(bert-base-uncased bert-large-uncased)

for model in ${models[@]}
do
    python blimp_lms.py --model ${model}
done