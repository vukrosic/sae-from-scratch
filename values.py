"""Runs every step once more and saves the numbers and curves the film draws: values.json."""
import contextlib
import glob
import io
import json

import numpy as np

import lm
import step1_model as s1
import step2_neurons as s2
import step3_superposition as s3
import step4_sparse_autoencoder as s4
import step5_sae_on_model as s5
import step6_features as s6
import step7_steering as s7
import step8_how_many as s8

R = lambda a, k=3: np.round(np.asarray(a, dtype=float), k).tolist()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    V = {'lang': {'topics': {k: [list(g) for g in v] for k, v in lm.TOPICS.items()}, 'names': lm.NAMES, 'vocab': len(lm.VOCAB),
                  'adj': lm.ADJ, 'sverb': lm.SVERB, 'func': lm.FUNC, 'patterns': lm.PATTERNS, 'kinds': s2.KINDS,
                  'n_ctx': lm.N_CTX, 'emb': lm.EMB, 'hid': lm.HID}}
    r = quiet(s1.run)
    model, words, topics, roles = lm.get_model()
    V['model'] = {'log': R(r['log']), 'loss': r['loss'], 'weights': r['weights'], 'blind': float(np.log(len(lm.VOCAB))),
                  'n_words': len(words), 'sample': ' '.join(lm.VOCAB[i] for i in words[:60]),
                  'writes': [lm.generate(model, p, 22, seed=3) for p in ('the sailor', 'the king', 'in the kitchen ,')],
                  'gram': lm.grammatical(lm.generate(model, 'the', 3000, seed=5))}
    h, ctx, nxt, topic, kind = s2.record(model, words, topics, roles)
    sae = s5.get_sae(h)
    f = sae.encode(h, s5.K)
    # one stretch of text, with every neuron and every feature at each word: the "busy vs sparse" picture
    a = 2000
    V['strip'] = {'words': [lm.VOCAB[i] for i in words[a + lm.N_CTX - 1:a + lm.N_CTX + 39]], 'topic': topic[a:a + 40].tolist(),
                  'h': R(h[a:a + 40], 2), 'f': [[(int(j), round(float(v), 2)) for j, v in enumerate(row) if v > 0] for row in f[a:a + 40]]}
    # the best guesses of the model at one position
    i = next(q for q in range(1000, 9000) if kind[q] == 1 and lm.VOCAB[ctx[q][-1]] == 'the' and lm.softmax(lm.scores(model, h[q:q + 1])).max() < 0.4)
    p = lm.softmax(lm.scores(model, h[i:i + 1]))[0]
    V['guess'] = {'text': [lm.VOCAB[w] for w in ctx[i]], 'next': lm.VOCAB[nxt[i]], 'top': [(lm.VOCAB[j], round(float(p[j]), 3)) for j in np.argsort(-p)[:6]],
                  'h': R(h[i], 1), 'f': [(int(j), round(float(v), 1)) for j, v in enumerate(f[i]) if v > 0], 'topic': lm.NAMES[topic[i]]}
    r = quiet(s2.run)
    V['neurons'] = {'by_topic': R(r['by_topic']), 'by_kind': R(r['by_kind']), 'on': R(r['on']), 'best': r['best'], 'n_top': r['n_top'],
                    'moments': r['moments'], 'on_at_once': float((h > 0).sum(1).mean()), 'positions': len(h)}
    r = quiet(s3.run)
    V['toy'] = {str(c): {'W': R(v['W']), 'b': R(v['b']), 'kept': v['kept'], 'error': v['error'], 'snaps': R(np.array(v['snaps'])[::3], 2)} for c, v in r.items()}
    r = quiet(s4.run)
    V['sae_toy'] = {'x': R(r['x'], 2), 'dirs': R(r['dirs']),
                    **{k.replace(' ', '_'): {'report': r[k]['report'], 'match': R(r[k]['match']), 'D': R(r[k]['sae'].D),
                                             'snaps': R(np.array(r[k]['snaps'])[::2], 2)} for k in ('no sparsity', 'sparse')}}
    r = quiet(s5.run)
    e = r['example']
    V['onmodel'] = {'n': s5.N_FEATURES, 'k': s5.K, 'steps': s5.STEPS, 'report': r['report'], 'swap': r['swap'], 'fires': R(r['fires'], 4),
                    'neuron_on': float(np.mean(h > 0)),
                    'example': {'h': R(e['h'], 1), 'on': [int(j) for j in e['on']], 'f': R(e['f'], 1), 'rebuilt': R(e['rebuilt'], 1), 'text': e['text'], 'next': e['next']}}
    r = quiet(s6.run)
    fe = r['features']
    V['features'] = {'by_topic': R(fe['by_topic']), 'by_kind': R(fe['by_kind']), 'names': fe['names'].tolist(), 'topic': fe['topic'].tolist(),
                     'kind': fe['kind'].tolist(), 'dense_by_topic': R(r['dense']['by_topic']), 'counts': r['counts'], 'topic_features': r['topic_features'],
                     'next_features': r['next_features'], 'next_words': {str(k): v for k, v in r['next_words'].items()}, 'moments': r['moments'],
                     'fires': R(r['fires'], 4), 'clear': s6.CLEAR}
    r = quiet(s7.run)
    V['steer'] = {'hero': s7.HERO, 'prompt': s7.PROMPT, 'plain': {'share': R(r['plain']['share']), 'gram': r['plain']['gram'], 'text': r['plain']['text']},
                  'topics': r['topics'], 'dial': r['dial'], 'story0': r['story0'], 'story1': r['story1'],
                  'neurons': [{'times': n['times'], 'reached': n['reached'], 'typical': float(np.median([q['share'] for q in n['rows']])),
                               'gram': float(np.median([q['gram'] for q in n['rows']]))} for n in r['neurons']]}
    V['sizes'] = quiet(s8.run)
    steps = sorted(glob.glob('step*.py'))
    V['code'] = {'steps': sum(len(open(p).read().splitlines()) for p in steps), 'all': sum(len(open(p).read().splitlines()) for p in glob.glob('*.py')),
                 'files': len(steps)}
    json.dump(V, open('values.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else o.tolist())
    print('values.json written,', len(json.dumps(V, default=lambda o: o.item() if hasattr(o, "item") else o.tolist())) // 1000, 'kB')


if __name__ == '__main__':
    main()
