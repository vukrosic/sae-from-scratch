"""The tiny language model whose mind we read, and the made-up language it learns.

The language: short sentences about 12 topics (the sea, a farm, a castle ...). Each topic has its own
people, things, places and verbs; a few words (the, a, big, sees ...) are shared by every topic.
The model: reads the last 8 words, guesses the next one. 8 word vectors -> 24 neurons -> 188 scores.
"""
import os

import numpy as np

np.seterr(divide='ignore', over='ignore', invalid='ignore')   # numpy 2.0 on Apple chips prints false matmul warnings

TOPICS = {
    'sea':      (['sailor', 'captain', 'whale', 'pirate'], ['ship', 'anchor', 'net', 'wave'], ['harbor', 'island', 'beach'], ['sails', 'rows', 'hauls']),
    'farm':     (['farmer', 'cow', 'horse', 'hen'], ['tractor', 'fence', 'hay', 'egg'], ['barn', 'meadow', 'orchard'], ['feeds', 'milks', 'plows']),
    'city':     (['driver', 'mayor', 'tourist', 'cop'], ['taxi', 'bus', 'sign', 'bridge'], ['street', 'station', 'square'], ['parks', 'drives', 'crosses']),
    'kitchen':  (['chef', 'baker', 'waiter', 'cook'], ['bread', 'soup', 'knife', 'pan'], ['kitchen', 'cafe', 'bakery'], ['bakes', 'stirs', 'slices']),
    'space':    (['astronaut', 'pilot', 'alien', 'robot'], ['rocket', 'comet', 'helmet', 'satellite'], ['moon', 'planet', 'crater'], ['launches', 'orbits', 'lands']),
    'school':   (['teacher', 'student', 'pupil', 'principal'], ['book', 'pencil', 'test', 'lesson'], ['classroom', 'library', 'school'], ['reads', 'grades', 'teaches']),
    'forest':   (['bear', 'wolf', 'owl', 'hunter'], ['tree', 'nest', 'log', 'mushroom'], ['forest', 'cave', 'river'], ['climbs', 'tracks', 'hunts']),
    'music':    (['singer', 'drummer', 'pianist', 'band'], ['song', 'guitar', 'drum', 'piano'], ['stage', 'studio', 'concert'], ['plays', 'sings', 'records']),
    'hospital': (['doctor', 'nurse', 'patient', 'surgeon'], ['pill', 'bandage', 'needle', 'chart'], ['hospital', 'clinic', 'ward'], ['heals', 'treats', 'examines']),
    'castle':   (['king', 'queen', 'knight', 'dragon'], ['sword', 'crown', 'shield', 'throne'], ['castle', 'tower', 'dungeon'], ['guards', 'rules', 'fights']),
    'sport':    (['player', 'coach', 'referee', 'goalkeeper'], ['ball', 'goal', 'whistle', 'trophy'], ['stadium', 'pitch', 'gym'], ['kicks', 'throws', 'wins']),
    'snow':     (['skier', 'snowman', 'penguin', 'climber'], ['sled', 'scarf', 'snowball', 'ice'], ['mountain', 'glacier', 'cabin'], ['skis', 'slides', 'shovels']),
}
NAMES = list(TOPICS)
ADJ = ['big', 'small', 'old', 'new', 'quiet', 'loud', 'happy', 'tired']
SVERB = ['sees', 'likes', 'finds', 'wants']
FUNC = ['the', 'a', 'and', 'in', 'near', 'is', ',', '.']
ROLES = ['who', 'what', 'where', 'does', 'adj', 'sverb', 'func']
VOCAB = FUNC + ADJ + SVERB + [w for t in TOPICS.values() for group in t for w in group]
ID = {w: i for i, w in enumerate(VOCAB)}
assert len(ID) == len(VOCAB) == 188
WORD_TOPIC = {w: k for k, name in enumerate(NAMES) for group in TOPICS[name] for w in group}   # word -> topic number
N_CTX, EMB, HID = 8, 12, 24
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.npz')

# every sentence follows one of five patterns; "adj?" is an adjective four times out of ten
PATTERNS = [
    'the adj? who does the adj? what .',
    'the who sverb a adj? what near the where .',
    'in the where , the who does a what .',
    'the what is adj and the where is adj .',
    'a who and a who sverb the where .',
]


def sentence(topic, rng):
    """One sentence about one topic, as (word, role) pairs."""
    who, what, where, does = TOPICS[NAMES[topic]]
    pools = {'who': who, 'what': what, 'where': where, 'does': does, 'adj': ADJ, 'sverb': SVERB}
    out = []
    for slot in PATTERNS[rng.integers(len(PATTERNS))].split():
        if slot == 'adj?':
            if rng.random() > 0.4:
                continue
            slot = 'adj'
        if slot in pools:
            out.append((pools[slot][rng.integers(len(pools[slot]))], slot))
        else:
            out.append((slot, 'func'))
    return out


def make_corpus(n_sentences=30000, seed=0):
    """A long stream of sentences. After each one the topic changes three times out of ten.
    Returns word numbers, the topic of each word's sentence, and each word's role."""
    rng = np.random.default_rng(seed)
    words, topics, roles = [], [], []
    topic = rng.integers(len(NAMES))
    for _ in range(n_sentences):
        if rng.random() < 0.3:
            topic = rng.integers(len(NAMES))
        for w, r in sentence(topic, rng):
            words.append(ID[w]); topics.append(topic); roles.append(ROLES.index(r))
    return np.array(words), np.array(topics), np.array(roles)


def windows(words):
    """Every stretch of 8 words in a row, and the word that came next."""
    idx = np.arange(N_CTX, len(words))
    return np.stack([words[idx - N_CTX + j] for j in range(N_CTX)], axis=1), words[idx]


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, b1, b2, eps, 0, None, None

    def step(self, params, grads):
        if self.m is None:
            self.m, self.v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def new_model(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.3, (len(VOCAB), EMB)),                      # E: one vector of 12 numbers per word
            rng.normal(0, np.sqrt(2 / (N_CTX * EMB)), (N_CTX * EMB, HID)), np.zeros(HID),   # W1, b1: into the 16 neurons
            rng.normal(0, np.sqrt(1 / HID), (HID, len(VOCAB))), np.zeros(len(VOCAB))]       # W2, b2: out to 188 scores


def neurons(model, ctx):
    """The 16 neuron activations after reading each row of 8 words. This is the 'mind' we will read."""
    E, W1, b1 = model[:3]
    return np.maximum(0, E[ctx].reshape(len(ctx), -1) @ W1 + b1)


def scores(model, h):
    return h @ model[3] + model[4]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def loss(model, ctx, nxt, hook=None):
    """Average surprise at the true next word. hook, if given, edits the neuron activations on the way."""
    h = neurons(model, ctx)
    if hook is not None:
        h = hook(h)
    p = softmax(scores(model, h))
    return float(-np.log(p[np.arange(len(nxt)), nxt] + 1e-12).mean())


def train(words, steps=6000, batch=256, lr=0.003, seed=0, log=None):
    model, opt, rng = new_model(seed), Adam(lr), np.random.default_rng(seed + 1)
    ctx, nxt = windows(words)
    for step in range(steps):
        i = rng.integers(0, len(ctx), batch)
        E, W1, b1, W2, b2 = model
        x = E[ctx[i]].reshape(batch, -1)
        h = np.maximum(0, x @ W1 + b1)
        p = softmax(h @ W2 + b2)
        if log is not None and step % 100 == 0:
            log.append(float(-np.log(p[np.arange(batch), nxt[i]]).mean()))
        dz = p; dz[np.arange(batch), nxt[i]] -= 1; dz /= batch       # backward pass, layer by layer
        dh = (dz @ W2.T) * (h > 0)
        dE = np.zeros_like(E)
        np.add.at(dE, ctx[i].ravel(), (dh @ W1.T).reshape(-1, EMB))
        opt.step(model, [dE, x.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)])
    return model


def get_model():
    """The trained model. Same seed every time, so every step studies the same network."""
    words, topics, roles = make_corpus()
    if not os.path.exists(CACHE):
        np.savez(CACHE, *train(words))
    with np.load(CACHE) as z:
        return [z[k] for k in z.files], words, topics, roles


def generate(model, prompt, n, seed=0, hook=None):
    """Continue a prompt, one sampled word at a time."""
    rng = np.random.default_rng(seed)
    out = [ID[w] for w in prompt.split()]
    for _ in range(n):
        ctx = np.array([([ID['.']] * N_CTX + out)[-N_CTX:]])
        h = neurons(model, ctx)
        if hook is not None:
            h = hook(h)
        p = softmax(scores(model, h))[0]
        out.append(int(rng.choice(len(p), p=p)))
    return ' '.join(VOCAB[i] for i in out)


def grammatical(text):
    """Share of the complete sentences in a text that follow one of the five patterns."""
    role = {w: 'func' for w in FUNC}
    role.update({w: 'adj' for w in ADJ}); role.update({w: 'sverb' for w in SVERB})
    for who, what, where, does in TOPICS.values():
        for group, r in ((who, 'who'), (what, 'what'), (where, 'where'), (does, 'does')):
            role.update({w: r for w in group})
    ok = set()
    for pat in PATTERNS:
        slots, variants = pat.split(), [[]]
        for s in slots:
            variants = [v + [s] for v in variants] if s != 'adj?' else [v + ['adj'] for v in variants] + variants
        ok |= {' '.join(v) for v in variants}
    sents = [s.split() + ['.'] for s in text.split(' . ')[1:-1] if s]
    good = sum(' '.join(w if role[w] == 'func' else role[w] for w in s) in ok for s in sents)
    return good / max(1, len(sents))
