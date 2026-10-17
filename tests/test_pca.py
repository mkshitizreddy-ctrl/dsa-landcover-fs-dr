import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from pca import PCA

rng = np.random.default_rng(1)
A = rng.standard_normal((600, 12)) @ rng.standard_normal((12, 12))
A[:, 3] += 2 * A[:, 0]

# 1. eigenvalues match numpy eigh on the correlation matrix
p = PCA(var_ratio=1.0, max_components=6).fit(A)
Z = (A - A.mean(0)) / A.std(0)
ref = np.sort(np.linalg.eigvalsh(np.cov(Z.T)))[::-1][:6]
assert np.allclose(p.values, ref, rtol=1e-4), (p.values, ref)

# 2. matches sklearn explained variance and |components|
from sklearn.decomposition import PCA as SK
sk = SK(n_components=6).fit(Z)
assert np.allclose(p.values, sk.explained_variance_, rtol=1e-4)
assert np.allclose(np.abs(p.components), np.abs(sk.components_), atol=1e-3)

# 3. components orthonormal
G = p.components @ p.components.T
assert np.allclose(G, np.eye(6), atol=1e-4)

# 4. variance-ratio stopping
q = PCA(var_ratio=0.95).fit(A)
assert q.explained_ratio.sum() >= 0.95
assert q.explained_ratio[:-1].sum() < 0.95
assert q.transform(A).shape == (600, len(q.values))
print("pca tests passed, k =", len(q.values))
