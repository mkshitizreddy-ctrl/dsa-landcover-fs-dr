import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "src")
from graph import build_corr_graph, bfs, dfs, components
from unionfind import UnionFind

# 1. data with a few correlated columns
rng = np.random.default_rng(1)
X = rng.normal(size=(300, 12))
X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=300)
X[:, 2] = X[:, 1] + 0.1 * rng.normal(size=300)
X[:, 5] = X[:, 4] + 0.1 * rng.normal(size=300)
C = np.abs(np.corrcoef(X.T))
t = 0.9
g = build_corr_graph(C, t)

# 2. edges match the threshold
for i in range(12):
    for j in range(12):
        if i != j:
            assert (j in g.adj[i]) == (C[i, j] > t)

# 3. BFS components match Union-Find and scipy
comp, k = components(g)
A = csr_matrix((C > t) & ~np.eye(12, dtype=bool))
k_ref, lab_ref = connected_components(A, directed=False)
assert k == k_ref
uf = UnionFind(12)
for i in range(12):
    for j in g.adj[i]:
        uf.union(i, j)
assert uf.count == k
for i in range(12):
    for j in range(12):
        same = comp[i] == comp[j]
        assert same == (uf.find(i) == uf.find(j)) == (lab_ref[i] == lab_ref[j])

# 4. DFS reaches the same nodes as BFS
for s in range(12):
    assert set(dfs(g, s)) == set(bfs(g, s))

# 5. depth-1 BFS gives exactly the neighbours
for s in range(12):
    assert set(bfs(g, s, max_depth=1)) - {s} == set(g.adj[s])

print("graph tests passed")