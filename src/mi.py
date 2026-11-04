from math import log

import numpy as np


def discretize(x, bins=10):
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros(len(x), dtype=int)
    b = ((x - lo) / (hi - lo) * bins).astype(int)
    b[b == bins] = bins - 1
    return b


def mutual_info(xb, y):
    n = len(xb)
    joint, px, py = {}, {}, {}
    for a, c in zip(xb, y):
        joint[(a, c)] = joint.get((a, c), 0) + 1
        px[a] = px.get(a, 0) + 1
        py[c] = py.get(c, 0) + 1
    mi = 0.0
    for (a, c), count in joint.items():
        mi += (count / n) * log(count * n / (px[a] * py[c]))
    return mi


def relevance_scores(X, y, bins=10):
    return [mutual_info(discretize(X[:, j], bins), y) for j in range(X.shape[1])]
