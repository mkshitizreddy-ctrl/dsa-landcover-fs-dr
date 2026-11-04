import numpy as np


class PCA:
    def __init__(self, var_ratio=0.95, max_components=None, iters=20000, tol=1e-12, seed=0):
        self.var_ratio = var_ratio
        self.max_components = max_components
        self.iters = iters
        self.tol = tol
        self.seed = seed

    def _top_eigenvector(self, C, rng):
        v = rng.standard_normal(C.shape[0])
        v /= np.linalg.norm(v)
        value = 0.0
        for _ in range(self.iters):
            w = C @ v
            norm = np.linalg.norm(w)
            if norm == 0:
                break
            w /= norm
            value = float(w @ C @ w)
            converged = np.linalg.norm(w - v) < self.tol
            v = w
            if converged:
                break
        return value, v

    def fit(self, X):
        n, d = X.shape
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = (X - self.mean) / self.std
        C = (Z.T @ Z) / (n - 1)
        total = np.trace(C)
        rng = np.random.default_rng(self.seed)
        values, vectors = [], []
        for _ in range(self.max_components or d):
            value, vector = self._top_eigenvector(C, rng)
            values.append(value)
            vectors.append(vector)
            C = C - value * np.outer(vector, vector)
            if self.max_components is None and sum(values) / total >= self.var_ratio:
                break
        self.values = np.array(values)
        self.components = np.array(vectors)
        self.explained_ratio = self.values / total
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std) @ self.components.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
