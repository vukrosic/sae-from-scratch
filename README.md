# Sparse autoencoders from scratch

Code for the video **How Researchers Read an AI's Mind — Sparse Autoencoders From Scratch**.
A tiny language model, the evidence that its neurons are mixed, a toy that shows why, and a sparse autoencoder that
finds readable features in it, which you can then turn like dials.
Python 3.9+ and numpy. No deep-learning library: the model, the autoencoder and their training loops are written out.

```bash
python3 tests.py      # checks every step, about 20 seconds
./run_all.sh          # runs all eight, about three minutes; expected output is in demo-output.txt
```

| File | What it builds | What the run shows |
| --- | --- | --- |
| `step1_model.py` | A next-word model: 8 words in, 24 neurons, 188 scores out (9,284 weights) | Trains in about 20 s. Surprise at the next word 5.16 → 1.03 (blind guessing: 5.24). 95.2 % of its sentences follow the grammar. |
| `step2_neurons.py` | A recording of the 24 neurons at 271,596 positions | A neuron is on at 63 % of positions, 15.1 are on at once. Most topic-minded neuron: 18 % of its activity in one topic (even spread: 8 %). Neurons with more than half their activity in one topic: 0 of 24. |
| `step3_superposition.py` | 5 features squeezed into 2 neurons and rebuilt | Features always on: 2 of 5 kept. On 10 % of the time: all 5 kept, arrows 72° apart (a pentagon). On 3 %: error 8.0 %. |
| `step4_sparse_autoencoder.py` | The sparse autoencoder (top-k), tested on step 3 where the answer is known | No sparsity: rebuilds 100.0 %, directions match the true arrows 0.896–0.994. One feature at a time: rebuilds 96.9 %, all five matches 0.998 or better. |
| `step5_sae_on_model.py` | The autoencoder on the model's neurons: 64 features, at most 3 on | Rebuilds 89.4 %. A feature is on at 4.7 % of positions. Swap test: surprise 1.019 (own neurons), 1.155 (rebuilt), 6.24 (erased): 97.4 % kept. |
| `step6_features.py` | What every feature responds to | Readable features (75 % of activity in one topic or one kind of next word): 16 name a topic (all 12 topics covered), 24 name the next word, 24 unclear. Neurons: 0 of 24. Without sparsity: 0 of 64. |
| `step7_steering.py` | Setting a feature while the model writes | Snow feature at its usual value: 97 % snow words, 91 % grammatical. At 3×: 100 % snow words, 0 % grammatical. 8 of 12 topic features steer to their topic (75 % or more). Single neurons: 0 of 12. |
| `step8_how_many.py` | Five dictionary sizes | See the table below. |

Dictionary size (step 8), at most 3 features on:

| Features | Rebuilds | Surprise after the swap | Dead | Topic | Next word | Unclear | Topics with a feature | Snow features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 75.0 % | 1.791 | 0 | 0 | 1 | 15 | 0 of 12 | 0 |
| 32 | 83.2 % | 1.363 | 0 | 1 | 7 | 24 | 1 of 12 | 1 |
| 64 | 89.4 % | 1.155 | 0 | 16 | 24 | 24 | 12 of 12 | 2 |
| 128 | 92.8 % | 1.075 | 1 | 39 | 58 | 30 | 12 of 12 | 4 |
| 256 | 94.5 % | 1.061 | 36 | 98 | 83 | 39 | 12 of 12 | 8 |

The model's own surprise is 1.019.

`lm.py` holds the language (188 words, 12 topics, five sentence patterns), the model, its training loop and the text
generator. `values.py` saves the numbers and curves the video draws. `model.npz` and `sae.npz` are written by steps 1
and 5 and read by the later steps.

## Exercise

In `step5_sae_on_model.py`, change `N_FEATURES, K, STEPS = 64, 3, 8000` to `64, 6, 8000`, then run steps 5, 6 and 7 again.
The neurons are rebuilt better. What happens to the number of features you can read?

Measured: rebuilds 93.7 % (was 89.4 %), surprise after the swap 1.091 (was 1.155). Readable features drop from
16 topic + 24 next word to 1 topic + 10 next word; 53 of 64 are unclear. The snow feature has 77 % of its activity in
snow (was 89 %), and holding it at its usual value gives 66 % snow words (was 97 %). Better rebuilding, fewer readable features.

## What is simplified

- **Language.** A made-up language of 188 words with five sentence patterns, so the true topics and word kinds are
  known and every claim can be counted.
- **Model.** One layer of 24 neurons that reads the last 8 words. Real language models are transformers with attention
  and many layers.
- **Autoencoder.** The top-k rule (keep the k strongest features), as in OpenAI's 2024 paper. The first sparse
  autoencoders used an L1 penalty. No dead-feature revival, no decoder tricks beyond unit-length directions.
- **"Readable".** A feature is readable when at least 75 % of its activity falls in one topic or before one kind of
  next word. That threshold is ours.
- **Steering.** Judged by the share of topic words that belong to the target topic and by a grammar checker for the
  five patterns. Four of the twelve topic features do not reach 75 %; the music feature reaches 11 %.
- **Dead.** A feature is dead when it never turns on over the recorded text.

## Licence

MIT for the code.
