import numpy as np
from kdtree import KDTree

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # 1. build the KD-tree on training points
        self.tree = KDTree(X)
        self.y = y
        return self

    def predict(self, Q):
        out = []
        for q in Q:
            votes = {}
            for _, i in self.tree.query(q, self.k):
                c = self.y[i]
                votes[c] = votes.get(c, 0) + 1
            # 2. majority vote, ties go to the smaller label
            out.append(max(votes, key=lambda c: (votes[c], -c)))
        return np.array(out)