"""Quick checks for every step. Run: python3 tests.py  (about 20 seconds; the first run also trains and saves the model)"""
import numpy as np

import lm
from step2_neurons import record, share_by
from step3_superposition import kept, train_toy
from step4_sparse_autoencoder import SAE, match, report, toy_neurons, train_sae
from step7_steering import clamp_feature, topic_share

# the language: 188 words, every sentence follows one of the five patterns
words, topics, roles = lm.make_corpus(2000, seed=3)
assert len(lm.VOCAB) == 188 and len(lm.NAMES) == 12
assert lm.grammatical(' '.join(lm.VOCAB[i] for i in words)) == 1.0

# the model: a few training steps already beat blind guessing
ctx, nxt = lm.windows(words)
assert lm.loss(lm.train(words, steps=300), ctx, nxt) < np.log(188) - 1

# step 2: shares of every unit add up to 1
acts = np.abs(np.random.default_rng(0).normal(size=(500, 6)))
assert np.allclose(share_by(acts, np.arange(500) % 4, 4).sum(0), 1)

# step 3: features that are always on -> 2 kept; rarely on -> all 5 kept, 72 degrees apart
assert kept(train_toy(1.0)[0]) == 2
W = train_toy(0.1)[0]
ang = np.sort(np.degrees(np.arctan2(W[:, 1], W[:, 0])) % 360)
assert kept(W) == 5 and np.abs(np.diff(ang) - 72).max() < 5

# step 4: "at most k on" really leaves k; the gradients agree with nudging a weight
rng = np.random.default_rng(1)
x = rng.normal(size=(64, 6))
sae = SAE(6, 20, seed=2, centre=x.mean(0))
assert ((sae.encode(x, 3) > 0).sum(1) <= 3).all()
f = sae.encode(x, 3)
err = sae.decode(f) - x
d_out = 2 * err / len(x)
d_f = (d_out @ sae.D.T) * (f > 0)
grads = [f.T @ d_out, (x - sae.c).T @ d_f, d_f.sum(0), d_out.sum(0) - (d_f @ sae.We.T).sum(0)]
base = (err ** 2).sum() / len(x)
for p, g, at in ((sae.D, grads[0], (int(np.nonzero(f.sum(0))[0][0]), 2)), (sae.We, grads[1], (1, int(np.nonzero(f.sum(0))[0][0]))), (sae.c, grads[3], (4,))):
    p[at] += 1e-6
    slope = (((sae.decode(sae.encode(x, 3)) - x) ** 2).sum() / len(x) - base) / 1e-6
    p[at] -= 1e-6
    assert abs(slope - g[at]) < 1e-3, (at, slope, g[at])

# step 4: the sparse autoencoder finds all 5 hidden directions of the toy
x, dirs = toy_neurons()
sae = train_sae(x, 8, 1)
assert match(sae, dirs, x).min() > 0.99 and report(sae, x)['active'] <= 1

# step 7: after clamping, the feature reads the value we set (when it is among the strongest)
x = np.abs(rng.normal(size=(50, 6)))
sae = train_sae(x, 12, 2, steps=200)
pushed = clamp_feature(sae, 5, 9.0)(x)
assert np.allclose((pushed - x) @ sae.D[5], 9.0 - sae.encode(x, 2)[:, 5])
assert topic_share('the whale sees a ship')[lm.NAMES.index('sea')] == 1.0

print('all tests pass')
