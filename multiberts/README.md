# Access the multi-berts

[Sellam et al. (2021)](https://arxiv.org/abs/2106.16163) released 25 different bert-base replications to facilitate robust analyses of the bert training procedure. Additionally for five of those models, they also released 28 different checkpoints at every 20000 steps upto the 200,000th step, and then after every 100,000 steps, giving us a whopping 140 different model checkpoints on which we can run analyses on. We use these 140 checkpoints for our BLiMP analyses.

Download the multi-bert models from the google storage space made available by Sellam et al. (2021):

```bash
for ckpt in {0..4} ; do
  wget "https://storage.googleapis.com/multiberts/public/intermediates/seed_${ckpt}.zip"
  unzip "seed_${ckpt}.zip"
done
```

Notice that these models are in the tensorflow format. The `transformers` library gives us a very convenient tool to convert tensorflow transformer models to pytorch models, easily accessible through minicons. To this end, run the following script:

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

lm.token_score(['The cat sat on the mat.'], prob = True)

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

## References

Sellam, T., Yadlowsky, S., Wei, J., Saphra, N., D'Amour, A., Linzen, T., ... & Pavlick, E. (2021). The multiberts: Bert reproductions for robustness analysis. arXiv preprint arXiv:2106.16163.


