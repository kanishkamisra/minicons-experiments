# Access the multi-berts

The multi-bert models used in this project involve the 5 bert-base models, each of which accompanied by 28 different checkpoints at every 20000 steps upto the 200,000th step, and then after every 100,000 steps, giving us a whopping 140 different models.

Download the multi-bert models from the google storage space made available by the authors of the original paper:

```bash
for ckpt in {0..4} ; do
  wget "https://storage.googleapis.com/multiberts/public/intermediates/seed_${ckpt}.zip"
  unzip "seed_${ckpt}.zip"
done
```

Notice that these models are in the tensorflow format. The transformers library gives us a very convenient tool to convert tensorflow transformer models to pytorch models, easily accessible through minicons. To this end, run the following script:

```bash
bash convert_models.sh
```

This results in the following directory structure:

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

To load this into minicons, use the following example, demonstrating how you can load one of the models (`seed_0/step_100000`):

```python
from minicons import scorer

lm = scorer.MaskedLMScorer('seed_0/step_100000')

lm.token_score(['The cat sat on the mat.'])

'''
[[('the', 0.4109024107456207),
  ('cat', 0.0017684451304376125),
  ('sat', 0.05783839523792267),
  ('on', 0.7951070070266724),
  ('the', 0.4369784891605377),
  ('mat', 0.0021842767018824816),
  ('.', 0.9733330011367798)]]
'''

```

Fin!


