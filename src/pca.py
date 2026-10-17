import numpy as np

class PCA:
    """PCA from scratch: covariance + power iteration with deflation."""

    def __init__(self, var_ratio=0.95, max_components=None, iters=20000, tol=1e-12, seed=0):
        self.var_ratio = var_ratio
        self.max_components = max_components
        self.iters = iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        n, d = X.shape
        # 1. centre and scale (z-score)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = (X - self.mean) / self.std
        # 2. covariance matrix, O(n d^2)
        C = (Z.T @ Z) / (n - 1)
        total = np.trace(C)
        rng = np.random.default_rng(self.seed)
        vals, vecs = [], []
        kmax = self.max_components or d
        # 3. power iteration with deflation, O(k * iters * d^2)
        for _ in range(kmax):
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            lam = 0.0
            for _ in range(self.iters):
                w = C @ v
                nw = np.linalg.norm(w)
                if nw == 0:
                    break
                w /= nw
                new_lam = float(w @ C @ w)
                done = np.linalg.norm(w - v) < self.tol
                v, lam = w, new_lam
                if done:
                    break
            vals.append(lam)
            vecs.append(v)
            C = C - lam * np.outer(v, v)      # deflate
            if sum(vals) / total >= self.var_ratio and self.max_components is None:
                break
        self.values = np.array(vals)
        self.components = np.array(vecs)          # k x d
        self.explained_ratio = self.values / total
        return self

    def transform(self, X):
        Z = (X - self.mean) / self.std
        return Z @ self.components.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
