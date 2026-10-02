import sys
import random
import numpy as np
from sklearn.metrics import mutual_info_score
sys.path.insert(0, "src")
from mi import discretize, mutual_info, relevance_scores
from sorting import merge_sort

# 1. merge sort matches Python sorted (including tie order)
a = [random.randint(0, 20) for _ in range(300)]
assert merge_sort(a) == sorted(a)
assert merge_sort(a, reverse=True) == sorted(a, reverse=True)
pairs = [(random.randint(0, 5), i) for i in range(100)]
assert merge_sort(pairs, key=lambda p: p[0], reverse=True) == sorted(pairs, key=lambda p: p[0], reverse=True)

# 2. our MI matches sklearn on the same bins
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 6))
y = (X[:, 0] + 0.5 * rng.normal(size=500) > 0).astype(int)
for j in range(6):
    xb = discretize(X[:, j], 10)
    assert abs(mutual_info(xb, y) - mutual_info_score(y, xb)) < 1e-9

# 3. the informative feature (column 0) ranks first
s = relevance_scores(X, y)
order = merge_sort(list(range(6)), key=lambda j: s[j], reverse=True)
assert order[0] == 0

print("mi and sorting tests passed")