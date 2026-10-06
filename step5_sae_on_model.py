"""Step 5. Train the sparse autoencoder on the real thing: the 24 neurons of the language model.

24 neuron values in, 64 features, at most 3 switched on at a time, 24 neuron values rebuilt.
Two checks: how much of the neurons' activity the features rebuild, and whether the model still
works when its real neurons are swapped for the rebuilt ones.
"""
import os

import numpy as np

import lm
from step2_neurons import record
from step4_sparse_autoencoder import SAE, report, train_sae

N_FEATURES, K, STEPS = 64, 3, 8000
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sae.npz')


def get_sae(h):
    """The trained dictionary. Same seed every time, so steps 6 and 7 study the same features."""
    if not os.path.exists(CACHE):
        np.savez(CACHE, *train_sae(h, N_FEATURES, K, STEPS).params())
    sae = SAE(h.shape[1], N_FEATURES)
    with np.load(CACHE) as z:
        sae.D, sae.We, sae.be, sae.c = [z[k] for k in z.files]
    sae.k = K
    return sae


def swap_test(model, sae, ctx, nxt, h_mean):
    """The model's surprise with its own neurons, with the rebuilt neurons, and with the neurons erased."""
    own = lm.loss(model, ctx, nxt)
    rebuilt = lm.loss(model, ctx, nxt, hook=lambda h: sae.decode(sae.encode(h, sae.k)))
    erased = lm.loss(model, ctx, nxt, hook=lambda h: np.zeros_like(h) + h_mean)
    return {'own': own, 'rebuilt': rebuilt, 'erased': erased, 'kept': (erased - rebuilt) / (erased - own)}


def run():
    model, words, topics, roles = lm.get_model()
    h, ctx, nxt, _, _ = record(model, words, topics, roles)
    sae = get_sae(h)
    r = report(sae, h)
    f = sae.encode(h, K)
    fires = (f > 0).mean(0)
    print(f'{h.shape[1]} neurons -> {N_FEATURES} features, at most {K} on at a time, {STEPS:,} training steps')
    print(f'rebuilds {r["explained"]:.1%} of the neurons\' activity;  features on per position {r["active"]:.2f};  dead features {r["dead"]}')
    print(f'a feature is on at {fires.mean():.1%} of positions (a neuron: {np.mean(h > 0):.0%});  busiest {fires.max():.1%}, rarest {fires.min():.2%}')
    s = swap_test(model, sae, ctx[:60000], nxt[:60000], h.mean(0))
    print(f'surprise at the next word:  own neurons {s["own"]:.3f}   rebuilt from features {s["rebuilt"]:.3f}   neurons erased {s["erased"]:.2f}')
    print(f'the features keep {s["kept"]:.1%} of what the neurons were doing for the model')
    i = 1000
    on = np.nonzero(f[i])[0]
    print('one position:', ' '.join(lm.VOCAB[w] for w in ctx[i]), ' -> ', lm.VOCAB[nxt[i]])
    print('  neurons :', ' '.join(f'{v:.1f}' for v in h[i]))
    print('  features:', ', '.join(f'#{j} = {f[i, j]:.1f}' for j in on), f' (the other {N_FEATURES - len(on)} are zero)')
    print('  rebuilt :', ' '.join(f'{v:.1f}' for v in sae.decode(f[i:i + 1])[0]))
    return {'report': r, 'swap': s, 'fires': fires, 'example': {'i': i, 'h': h[i], 'on': on, 'f': f[i, on], 'rebuilt': sae.decode(f[i:i + 1])[0],
                                                           'text': ' '.join(lm.VOCAB[w] for w in ctx[i]), 'next': lm.VOCAB[nxt[i]]}}


if __name__ == '__main__':
    run()
