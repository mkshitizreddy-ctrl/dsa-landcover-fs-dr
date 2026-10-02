import sys
import numpy as np
sys.path.insert(0, "src")
from graph import build_corr_graph
from selection import select_features, select_naive, select_per_component

# 1. data with correlated pairs
rng = np.random.default_rng(2)
X = rng.normal(size=(400, 20))
for j in range(1, 20, 2):
    X[:, j] = X[:, j - 1] + 0.2 * rng.normal(size=400)
C = np.abs(np.corrcoef(X.T))
C = np.triu(C) + np.triu(C, 1).T
scores = list(rng.random(20))

for t in (0.5, 0.8, 0.9):
    g = build_corr_graph(C, t)
    S = select_features(scores, g)
    # 2. heap + BFS gives the same subset as the naive greedy
    assert sorted(S) == sorted(select_naive(scores, C, t))
    # 3. no two selected features are correlated above t
    for a in S:
        for b in S:
            if a != b:
                assert C[a, b] <= t
    # 4. every dropped feature has a selected neighbour
    for f in range(20):
        if f not in S:
            assert any(C[f, s] > t for s in S)
    # 5. per-component variant never selects more
    assert len(select_per_component(scores, g)) <= len(S)

print("selection tests passed")