import argparse
from transformers import BertTokenizer

tokenizer = BertTokenizer(vocab_file = 'vocab.txt')

parser = argparse.ArgumentParser()

parser.add_argument(
        '--target',
        '-t',
        type=str,
        required=True
)

args = parser.parse_args()

tokenizer.save_pretrained(args.target)
