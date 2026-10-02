from heap import IndexedMaxHeap
from graph import bfs
from unionfind import UnionFind
from sorting import merge_sort

def select_features(scores, g):
    # 1. max-heap keyed by relevance, O(d) build
    H = IndexedMaxHeap.build(scores)
    removed = set()
    S = []
    while len(H):
        f, _ = H.pop()          # 2. best remaining, O(log d)
        if f in removed:
            continue
        S.append(f)
        # 3. drop f's direct neighbours (BFS depth 1)
        for v in bfs(g, f, max_depth=1):
            if v != f:
                removed.add(v)
    return S

def select_per_component(scores, g):
    # 4. variant: best-scoring feature in each connected component
    uf = UnionFind(g.n)
    for u in range(g.n):
        for v in g.adj[u]:
            uf.union(u, v)
    best = {}
    for f in range(g.n):
        r = uf.find(f)
        if r not in best or scores[f] > scores[best[r]]:
            best[r] = f
    return sorted(best.values())

def select_naive(scores, C, t):
    # 5. baseline: sort by score, keep f if no kept feature is correlated, O(d^2)
    order = merge_sort(list(range(len(scores))), key=lambda j: scores[j], reverse=True)
    S = []
    for f in order:
        if all(C[f][s] <= t for s in S):
            S.append(f)
    return S