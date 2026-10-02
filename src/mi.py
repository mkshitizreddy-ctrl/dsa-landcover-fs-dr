import numpy as np
from math import log

def discretize(x, bins=10):
    # 1. equal-width bins, ids 0..bins-1
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros(len(x), dtype=int)
    b = ((x - lo) / (hi - lo) * bins).astype(int)
    b[b == bins] = bins - 1
    return b

def mutual_info(xb, y):
    # 2. I(X;Y) from frequency counts, in nats
    n = len(xb)
    joint, px, py = {}, {}, {}
    for a, c in zip(xb, y):
        joint[(a, c)] = joint.get((a, c), 0) + 1
        px[a] = px.get(a, 0) + 1
        py[c] = py.get(c, 0) + 1
    mi = 0.0
    for (a, c), cnt in joint.items():
        mi += (cnt / n) * log(cnt * n / (px[a] * py[c]))
    return mi

def relevance_scores(X, y, bins=10):
    # 3. MI of every column with the label
    return [mutual_info(discretize(X[:, j], bins), y) for j in range(X.shape[1])]