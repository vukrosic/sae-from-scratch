"""Step 2. Open the model and watch its 24 neurons. Does any neuron mean one thing?

For every position in the text we record the 24 neuron values, the topic of the text there,
and the kind of word that comes next. Then we ask each neuron: where does your activity go?
"""
import numpy as np

import lm

KINDS = ['a person', 'a thing', 'a place', 'an action', 'a describing word', 'a liking word', 'a small word']


def record(model, words, topics, roles):
    """Neuron values at every position, with the topic there and the kind of the next word."""
    ctx, nxt = lm.windows(words)
    return lm.neurons(model, ctx), ctx, nxt, np.asarray(topics)[lm.N_CTX - 1:-1], np.asarray(roles)[lm.N_CTX:]


def share_by(acts, labels, n_labels):
    """For each unit (column): how its average activity is split between the labels. Rows = labels, columns sum to 1."""
    mean = np.stack([acts[labels == k].mean(0) for k in range(n_labels)])
    return mean / np.maximum(mean.sum(0, keepdims=True), 1e-9)


def show(ctx, nxt, i):
    return ' '.join(lm.VOCAB[w] for w in ctx[i]) + '  ->  ' + lm.VOCAB[nxt[i]]


def run():
    model, words, topics, roles = lm.get_model()
    h, ctx, nxt, topic, kind = record(model, words, topics, roles)
    by_topic, by_kind = share_by(h, topic, len(lm.NAMES)), share_by(h, kind, len(lm.ROLES))
    print(f'recorded {len(h):,} positions x {h.shape[1]} neurons;  a neuron is on at {np.mean(h > 0):.0%} of positions, '
          f'{np.mean((h > 0).sum(1)):.1f} neurons are on at once')
    print('neuron   on    strongest topic      strongest next-word kind')
    for j in range(h.shape[1]):
        t, k = by_topic[:, j].argmax(), by_kind[:, j].argmax()
        print(f'  {j:>2}   {np.mean(h[:, j] > 0):>4.0%}   {lm.NAMES[t]:<9} {by_topic[t, j]:>4.0%}       {KINDS[k]:<18} {by_kind[k, j]:>4.0%}')
    best = by_topic.max(0)
    print(f'most topic-minded neuron: {best.max():.0%} of its activity in one topic (an even spread over 12 topics is 8%)')
    print(f'most kind-minded neuron: {by_kind.max():.0%} of its activity before one kind of word (an even spread over 7 kinds is 14%)')
    print(f'neurons with more than half of their activity in one topic: {(best > 0.5).sum()} of {h.shape[1]}')
    j = int(best.argmax())
    top = np.argsort(-h[:, j])[::400][:8]
    print(f'neuron {j}, eight of its strongest moments:')
    for i in top:
        print(f'   {h[i, j]:.1f}  [{lm.NAMES[topic[i]]}]  {show(ctx, nxt, i)}')
    n_top = [len(set(topic[np.argsort(-h[:, q])[:200]])) for q in range(h.shape[1])]
    print(f'topics among each neuron\'s 200 strongest moments: from {min(n_top)} to {max(n_top)} of 12, typical {int(np.median(n_top))}')
    return {'by_topic': by_topic, 'by_kind': by_kind, 'on': (h > 0).mean(0), 'best': j, 'n_top': n_top,
            'moments': [(float(h[i, j]), lm.NAMES[topic[i]], show(ctx, nxt, i)) for i in top]}


if __name__ == '__main__':
    run()
