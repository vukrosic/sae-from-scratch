"""Step 4. The sparse autoencoder, tested where we know the answer.

Take the 2 neurons of the toy from step 3. We know 5 features are hidden in them, and in which directions.
The autoencoder sees only the 2 neuron values. It has to rebuild them from a dictionary of directions,
using one direction at a time. Does it find the 5 hidden ones?
"""
import numpy as np

from lm import Adam
from step3_superposition import data, train_toy

np.seterr(divide='ignore', over='ignore', invalid='ignore')


class SAE:
    """x -> features -> x again.  features = relu((x - c) @ We + be),  rebuilt = features @ D + c."""

    def __init__(self, d, n, seed=0, centre=None):
        rng = np.random.default_rng(seed)
        D = rng.normal(size=(n, d))
        self.D = D / np.linalg.norm(D, axis=1, keepdims=True)    # the dictionary: one direction of length 1 per feature
        self.We = self.D.T.copy()                                # the detector for each feature starts as its own direction
        self.be = np.zeros(n)
        self.c = np.zeros(d) if centre is None else centre.copy()   # the centre of the data

    def encode(self, x, k=None):
        f = np.maximum(0, (x - self.c) @ self.We + self.be)
        if k is not None:                                        # sparse: only the k strongest features may speak
            f = f * (f >= np.partition(f, -k, axis=1)[:, -k][:, None])
        return f

    def decode(self, f):
        return f @ self.D + self.c

    def params(self):
        return [self.D, self.We, self.be, self.c]


def train_sae(x, n, k, steps=4000, batch=256, lr=0.003, seed=0, snapshots=None):
    """Loss for one vector: the squared distance between x and the rebuilt x. k = None means no sparsity."""
    sae, opt, rng = SAE(x.shape[1], n, seed, x.mean(0)), Adam(lr), np.random.default_rng(seed + 1)
    sae.k = k
    for step in range(steps):
        xb = x[rng.integers(0, len(x), batch)]
        f = sae.encode(xb, k)
        err = sae.decode(f) - xb
        d_out = 2 * err / batch                                  # backward pass
        d_f = (d_out @ sae.D.T) * (f > 0)
        grads = [f.T @ d_out, (xb - sae.c).T @ d_f, d_f.sum(0), d_out.sum(0) - (d_f @ sae.We.T).sum(0)]
        opt.step(sae.params(), grads)
        sae.D /= np.linalg.norm(sae.D, axis=1, keepdims=True)    # keep every direction at length 1
        if snapshots is not None and step % 20 == 0:
            snapshots.append(sae.D.copy())
    return sae


def report(sae, x):
    """How well it rebuilds, and how many features it uses for one vector."""
    f = sae.encode(x, sae.k)
    err = ((sae.decode(f) - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    return {'explained': float(1 - err), 'active': float((f > 0).sum(1).mean()), 'dead': int(((f > 0).sum(0) == 0).sum())}


def toy_neurons(n=20000, seed=5):
    """The 2 neuron values of the step-3 toy on fresh inputs, and the 5 directions hidden in them."""
    W, _ = train_toy(0.1)
    x = data(n, 0.1, np.random.default_rng(seed)) @ W
    return x, W / np.linalg.norm(W, axis=1, keepdims=True)


def match(sae, dirs, x):
    """For each hidden direction, how closely the best live feature points along it (1 = exactly)."""
    alive = (sae.encode(x, sae.k) > 0).mean(0) > 0.001
    return (dirs @ sae.D[alive].T).max(axis=1)


def run():
    x, dirs = toy_neurons()
    out = {'x': x[:400], 'dirs': dirs}
    for name, k in (('no sparsity', None), ('sparse', 1)):
        snaps = []
        sae = train_sae(x, 8, k, snapshots=snaps)
        r, m = report(sae, x), match(sae, dirs, x)
        print(f'{name:<12} rebuilt {r["explained"]:.1%}   features used per vector {r["active"]:.2f}   '
              f'match to the 5 hidden directions: {" ".join(f"{v:.3f}" for v in m)}')
        out[name] = {'report': r, 'match': m.tolist(), 'snaps': snaps, 'sae': sae}
    return out


if __name__ == '__main__':
    run()
