"""Step 1. A tiny language model: read 8 words, guess the next. This is the network we open up."""
import time

import numpy as np

import lm


def run():
    words, topics, roles = lm.make_corpus()
    print(f'language: {len(lm.VOCAB)} words, {len(lm.NAMES)} topics, {len(words):,} words of text')
    print('sample:', ' '.join(lm.VOCAB[i] for i in words[:40]))
    log, t0 = [], time.time()
    model = lm.train(words, log=log)
    n = sum(p.size for p in model)
    print(f'model: {n:,} weights, {lm.HID} neurons in the middle, trained in {time.time() - t0:.0f} s')
    test, _, _ = lm.make_corpus(3000, seed=1)
    ctx, nxt = lm.windows(test)
    L = lm.loss(model, ctx, nxt)
    print(f'surprise at the next word, before training {log[0]:.2f}, after {L:.2f}  (guessing blindly: {np.log(len(lm.VOCAB)):.2f})')
    for prompt in ('the sailor', 'the king', 'in the kitchen ,'):
        print('writes:', lm.generate(model, prompt, 22, seed=3))
    text = lm.generate(model, 'the', 3000, seed=5)
    print(f'grammatical sentences in 3,000 generated words: {lm.grammatical(text):.1%}')
    return {'log': log, 'loss': L, 'weights': n}


if __name__ == '__main__':
    run()
