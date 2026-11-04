from graph import bfs
from heap import IndexedMaxHeap
from sorting import merge_sort
from unionfind import UnionFind


def select_features(scores, g):
    heap = IndexedMaxHeap.build(scores)
    removed = set()
    selected = []
    while len(heap):
        f, _ = heap.pop()
        if f in removed:
            continue
        selected.append(f)
        for v in bfs(g, f, max_depth=1):
            if v != f:
                removed.add(v)
    return selected


def select_per_component(scores, g):
    uf = UnionFind(g.n)
    for u in range(g.n):
        for v in g.adj[u]:
            uf.union(u, v)
    best = {}
    for f in range(g.n):
        root = uf.find(f)
        if root not in best or scores[f] > scores[best[root]]:
            best[root] = f
    return sorted(best.values())


def select_naive(scores, C, t):
    order = merge_sort(list(range(len(scores))), key=lambda j: scores[j], reverse=True)
    selected = []
    for f in order:
        if all(abs(C[f][s]) <= t for s in selected):
            selected.append(f)
    return selected
