"""Step 8. How many features should we ask for? Train dictionaries of five sizes and compare.

A bigger dictionary rebuilds the neurons better. It also splits features: one "snow" feature
becomes several, each for snow in a different situation. And some features never switch on at all.
"""
from multiprocessing import Pool

import numpy as np

import lm
from step2_neurons import KINDS, record
from step4_sparse_autoencoder import report, train_sae
from step5_sae_on_model import K, STEPS, swap_test
from step6_features import CLEAR, count, read

SIZES = [16, 32, 64, 128, 256]
N_EVAL = 60000


def one(n):
    model, words, topics, roles = lm.get_model()
    h, ctx, nxt, topic, kind = record(model, words, topics, roles)
    sae = train_sae(h, n, K, STEPS)
    h, ctx, nxt, topic, kind = h[:N_EVAL], ctx[:N_EVAL], nxt[:N_EVAL], topic[:N_EVAL], kind[:N_EVAL]
    f = sae.encode(h, K)
    alive = (f > 0).mean(0) > 0
    r = read(f[:, alive], topic, kind)
    snow = lm.NAMES.index('snow')
    family = [(float(r['by_topic'][snow, j]), KINDS[r['kind'][j]], float(r['by_kind'][r['kind'][j], j]))
              for j in range(alive.sum()) if r['topic'][j] == snow and r['by_topic'][snow, j] >= CLEAR]
    per_topic = [int(((r['topic'] == t) & (r['by_topic'].max(0) >= CLEAR)).sum()) for t in range(len(lm.NAMES))]
    return {'n': n, 'explained': report(sae, h)['explained'], 'dead': int(n - alive.sum()), 'counts': count(r),
            'swap': swap_test(model, sae, ctx, nxt, h.mean(0)), 'snow': sorted(family, reverse=True), 'per_topic': per_topic}


def run():
    lm.get_model()                                   # make sure the cached model exists before the workers start
    with Pool(len(SIZES)) as pool:
        rows = pool.map(one, SIZES)
    print('features   rebuilds   surprise (own 1.02)   dead   name a topic   name the next word   unclear   topics with a feature')
    for r in rows:
        c = r['counts']
        print(f'  {r["n"]:>4}     {r["explained"]:>6.1%}        {r["swap"]["rebuilt"]:.3f}          {r["dead"]:>3}       {c["topic"]:>3}'
              f'              {c["next word"]:>3}            {c["unclear"]:>3}            {sum(v > 0 for v in r["per_topic"]):>2} of 12')
    for r in rows:
        print(f'{r["n"]:>4} features: {len(r["snow"])} snow feature(s): ' + '; '.join(f'{p:.0%} snow, mostly before {k} ({q:.0%})' for p, k, q in r['snow']))
    return rows


if __name__ == '__main__':
    run()
