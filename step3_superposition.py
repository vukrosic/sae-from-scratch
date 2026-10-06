"""Step 3. Why neurons get mixed: more things to remember than neurons to remember them with.

A toy with a known answer. The input has 5 separate features. The network must push them through
only 2 neurons and rebuild all 5 on the other side:

    neurons = x @ W            (5 numbers -> 2 numbers)
    rebuilt = relu(neurons @ W.T + b)

When every feature is on all the time, it keeps 2 and drops 3. When features are rarely on,
it keeps all 5: each one gets its own direction across the 2 neurons.
"""
import numpy as np

from lm import Adam

N_FEAT, N_NEURON = 5, 2


def data(n, chance, rng):
    """Each feature is on with the given chance; when on, its value is between 0 and 1."""
    return rng.uniform(0, 1, (n, N_FEAT)) * (rng.random((n, N_FEAT)) < chance)


def train_toy(chance, steps=6000, batch=512, lr=0.01, seed=0, snapshots=None):
    rng = np.random.default_rng(seed)
    W, b, opt = rng.normal(0, 0.3, (N_FEAT, N_NEURON)), np.zeros(N_FEAT), Adam(lr)
    for step in range(steps):
        x = data(batch, chance, rng)
        h = x @ W
        out = np.maximum(0, h @ W.T + b)
        d_out = 2 * (out - x) * (out > 0) / batch
        d_h = d_out @ W
        opt.step([W, b], [x.T @ d_h + d_out.T @ h, d_out.sum(0)])
        if snapshots is not None and step % 20 == 0:
            snapshots.append(W.copy())
    return W, b


def kept(W):
    """A feature is kept when its direction has real length."""
    return int((np.linalg.norm(W, axis=1) > 0.5).sum())


def error(W, b, chance, seed=9):
    x = data(20000, chance, np.random.default_rng(seed))
    out = np.maximum(0, x @ W @ W.T + b)
    return float(((out - x) ** 2).sum() / (x ** 2).sum())


def run():
    out = {}
    for chance in (1.0, 0.3, 0.1, 0.03):
        snaps = []
        W, b = train_toy(chance, snapshots=snaps)
        ang = np.sort(np.degrees(np.arctan2(W[:, 1], W[:, 0])) % 360)
        gaps = np.diff(np.append(ang, ang[0] + 360))
        print(f'feature on {chance:>4.0%} of the time: kept {kept(W)} of {N_FEAT}   lengths {" ".join(f"{v:.2f}" for v in np.linalg.norm(W, axis=1))}'
              f'   gaps between directions {" ".join(f"{g:.0f}" for g in gaps)}   error {error(W, b, chance):.1%}')
        out[chance] = {'W': W, 'b': b, 'snaps': snaps, 'kept': kept(W), 'error': error(W, b, chance)}
    return out


if __name__ == '__main__':
    run()
