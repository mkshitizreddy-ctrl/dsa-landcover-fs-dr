import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, "src")
from kdtree import KDTree
from knn import KNN

# 1. three shifted gaussian classes
rng = np.random.default_rng(3)
X = rng.normal(size=(600, 5)) + np.repeat(np.arange(3), 200)[:, None]
y = np.repeat(np.arange(3), 200)
Q = rng.normal(size=(100, 5)) + rng.integers(0, 3, size=(100, 1))

# 2. KD-tree distances equal brute force
tree = KDTree(X)
for q in Q:
    got = [d for d, _ in tree.query(q, 5)]
    ref = np.sort(((X - q) ** 2).sum(axis=1))[:5]
    assert np.allclose(got, ref)

# 3. predictions match sklearn
mine = KNN(5).fit(X, y).predict(Q)
ref = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X, y).predict(Q)
assert (mine == ref).all()

print("kdtree and knn tests passed")