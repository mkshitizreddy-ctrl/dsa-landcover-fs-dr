import numpy as np

from kdtree import KDTree


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.tree = KDTree(X)
        self.y = y
        return self

    def predict_one(self, q):
        votes = {}
        for _, i in self.tree.query(q, self.k):
            votes[self.y[i]] = votes.get(self.y[i], 0) + 1
        most = max(votes.values())
        return min(label for label, count in votes.items() if count == most)

    def predict(self, Q):
        return np.array([self.predict_one(q) for q in Q])
