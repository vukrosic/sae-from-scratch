"""Step 7. Turn a feature up by hand and watch the model change what it writes.

While the model writes, at every word we read its 24 neurons, find the features, set one feature
to a value we choose, and write the changed neurons back:

    neurons = neurons + (value - feature) * that feature's direction
"""
import numpy as np

import lm
from step2_neurons import record
from step5_sae_on_model import K, get_sae
from step6_features import read, topic_features

N_WORDS = 2000
HERO = 'snow'            # the topic we turn up in the detailed test
PROMPT = 'in the kitchen , the chef'


def clamp_feature(sae, j, value):
    def hook(h):
        f = sae.encode(h, sae.k)
        return h + (value - f[:, j:j + 1]) * sae.D[j]
    return hook


def clamp_neuron(j, value):
    def hook(h):
        h = h.copy()
        h[:, j] = value
        return h
    return hook


def topic_share(text):
    """Of the topic words in a text, the share that belongs to each topic."""
    t = [lm.WORD_TOPIC[w] for w in text.split() if w in lm.WORD_TOPIC]
    return np.bincount(t, minlength=len(lm.NAMES)) / max(len(t), 1)


def write(model, hook, n=N_WORDS, seed=1):
    text = lm.generate(model, 'the', n, seed=seed, hook=hook)
    return text, topic_share(text), lm.grammatical(text)


def run():
    model, words, topics, roles = lm.get_model()
    h, ctx, nxt, topic, kind = record(model, words, topics, roles)
    sae = get_sae(h)
    f = sae.encode(h, K)
    tf = topic_features(read(f, topic, kind), f)
    usual = np.array([f[f[:, j] > 0, j].mean() for j in range(f.shape[1])])     # a feature's usual value when it is on
    text, share, gram = write(model, None)
    print(f'untouched: largest topic share {share.max():.0%}, grammatical {gram:.0%}')
    print('   ', ' '.join(text.split()[:28]))
    out = {'plain': {'share': share, 'gram': gram, 'text': text[:400]}, 'topics': [], 'dial': [], 'neurons': []}
    print('every topic feature held at its usual value, at every word:')
    for t, j in enumerate(tf):
        text, share, gram = write(model, clamp_feature(sae, j, usual[j]))
        out['topics'].append({'topic': lm.NAMES[t], 'feature': j, 'share': float(share[t]), 'gram': gram, 'text': ' '.join(text.split()[40:68])})
        print(f'  {lm.NAMES[t]:<9} #{j:<2} {share[t]:>4.0%} of topic words are {lm.NAMES[t]} words, grammatical {gram:>4.0%}   ' + ' '.join(text.split()[40:54]))
    t = lm.NAMES.index(HERO)
    j = tf[t]
    print(f'the {HERO} feature (#{j}, usual value {usual[j]:.1f}) at different strengths:')
    for m in (0, 0.5, 1, 1.5, 2, 3, 5, 10):
        text, share, gram = write(model, clamp_feature(sae, j, m * usual[j]))
        out['dial'].append({'times': m, 'share': float(share[t]), 'gram': gram, 'text': ' '.join(text.split()[40:68])})
        print(f'  x{m:<4} {HERO} words {share[t]:>4.0%}   grammatical {gram:>4.0%}   ' + ' '.join(text.split()[40:56]))
    for m in (0, 1):
        text = lm.generate(model, PROMPT, 30, seed=2, hook=clamp_feature(sae, j, m * usual[j]) if m else None)
        out['story' + str(m)] = text
        print(f'  prompt "{PROMPT}", {HERO} feature {"on" if m else "untouched"}:  {text}')
    reach = sum(r['share'] >= 0.75 for r in out['topics'])
    print(f'features: {reach} of 12 topics reached (at least 75% of topic words), and we knew which feature to turn')
    for m in (1, 3):
        rows = []
        for q in range(h.shape[1]):
            text, share, gram = write(model, clamp_neuron(q, m * h[h[:, q] > 0, q].mean()), n=1000)
            rows.append({'neuron': q, 'top': lm.NAMES[int(share.argmax())], 'share': float(share.max()), 'gram': gram})
        got = sorted({r['top'] for r in rows if r['share'] >= 0.75})
        print(f'single neurons held at {m} x their usual value: {len(got)} of 12 topics reached ({", ".join(got) or "none"});  '
              f'typical neuron: {np.median([r["share"] for r in rows]):.0%} for its strongest topic, grammatical {np.median([r["gram"] for r in rows]):.0%}')
        out['neurons'].append({'times': m, 'rows': rows, 'reached': got})
    return out


if __name__ == '__main__':
    run()
