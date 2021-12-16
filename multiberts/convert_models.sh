#!/bin/bash


for ckpt in {0..4}; do
        for file in seed_${ckpt}/*; do
                transformers-cli convert --model_type bert --tf_checkpoint $file/bert.ckpt --config bert_config.json --pytorch_dump_output $file/pytorch_model.bin;
                python save_tokenizer.py -t $file;
                rm $file/bert.ckpt.data-00000-of-00001;
                cp bert_config.json $file/config.json
        done
done
        #transformers-cli convert --model_type bert --tf
