"""Step 6. Read the features. Does a feature mean one thing where a neuron did not?

Same question as step 2, asked of the 64 features: where does your activity go?
A unit "names a topic" when at least 3/4 of its activity is in one topic.
A unit "names the next word" when at least 3/4 of its activity comes before one kind of word.
"""
import numpy as np

import lm
from step2_neurons import KINDS, record, share_by, show
from step4_sparse_autoencoder import train_sae
from step5_sae_on_model import K, N_FEATURES, STEPS, get_sae

CLEAR = 0.75


def read(acts, topic, kind):
    """For every unit: its strongest topic, its strongest next-word kind, and what it names (if anything)."""
    by_topic, by_kind = share_by(acts, topic, len(lm.NAMES)), share_by(acts, kind, len(lm.ROLES))
    names = np.where(by_topic.max(0) >= CLEAR, 'topic', np.where(by_kind.max(0) >= CLEAR, 'next word', 'unclear'))
    return {'by_topic': by_topic, 'by_kind': by_kind, 'names': names,
            'topic': by_topic.argmax(0), 'kind': by_kind.argmax(0)}


def count(r):
    return {what: int((r['names'] == what).sum()) for what in ('topic', 'next word', 'unclear')}


def next_words(f_col, nxt, n=3):
    w = np.bincount(nxt, weights=f_col, minlength=len(lm.VOCAB))
    w /= w.sum()
    return [(lm.VOCAB[i], float(w[i])) for i in np.argsort(-w)[:n]]


def topic_features(r, f):
    """For each topic, the feature that names it best."""
    return [int(r['by_topic'][t].argmax()) for t in range(len(lm.NAMES))]


def run():
    model, words, topics, roles = lm.get_model()
    h, ctx, nxt, topic, kind = record(model, words, topics, roles)
    sae = get_sae(h)
    f = sae.encode(h, K)
    dense = train_sae(h, N_FEATURES, None, STEPS)
    fd = dense.encode(h)
    rn, rd, rf = read(h, topic, kind), read(fd, topic, kind), read(f, topic, kind)
    print(f'what each unit names (at least {CLEAR:.0%} of its activity in one place):')
    for label, r, n in (('24 neurons', rn, 24), ('64 features, no sparsity', rd, N_FEATURES), ('64 sparse features', rf, N_FEATURES)):
        c = count(r)
        print(f'  {label:<26} a topic {c["topic"]:>2}   the kind of next word {c["next word"]:>2}   unclear {c["unclear"]:>2}')
    tf = topic_features(rf, f)
    print('the best feature for each topic:')
    for t, j in enumerate(tf):
        print(f'  {lm.NAMES[t]:<9} feature #{j:<2}  {rf["by_topic"][t, j]:>4.0%} of its activity is in {lm.NAMES[t]}   on at {np.mean(f[:, j] > 0):.1%} of positions')
    print('features that name the next word:')
    shown = []
    for j in np.argsort(-f.sum(0)):
        if rf['names'][j] == 'next word' and len(shown) < 10:
            nw = next_words(f[:, j], nxt)
            shown.append(int(j))
            print(f'  #{j:<2} {KINDS[rf["kind"][j]]:<18} {rf["by_kind"][rf["kind"][j], j]:>4.0%}   next: ' + ', '.join(f'{w} {p:.0%}' for w, p in nw))
    j = tf[0]
    top = np.argsort(-f[:, j])[::300][:8]
    print(f'feature #{j}, eight of its strongest moments:')
    for i in top:
        print(f'   {f[i, j]:.1f}  [{lm.NAMES[topic[i]]}]  {show(ctx, nxt, i)}')
    return {'neurons': rn, 'dense': rd, 'features': rf, 'counts': [count(rn), count(rd), count(rf)], 'topic_features': tf,
            'next_features': shown, 'next_words': {j: next_words(f[:, j], nxt) for j in shown},
            'moments': [(float(f[i, j]), lm.NAMES[topic[i]], show(ctx, nxt, i)) for i in top], 'fires': (f > 0).mean(0)}


if __name__ == '__main__':
    run()
