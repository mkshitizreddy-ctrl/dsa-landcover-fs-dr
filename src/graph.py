class Graph:
    # 1. adjacency list, undirected
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

def build_corr_graph(C, t):
    # 2. edge between features i, j if |corr| > t, O(d^2)
    d = len(C)
    g = Graph(d)
    for i in range(d):
        for j in range(i + 1, d):
            if C[i][j] > t:
                g.add_edge(i, j)
    return g

def bfs(g, s, max_depth=None):
    # 3. queue as list + head index, returns {node: distance}, O(V + E)
    dist = {s: 0}
    q = [s]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        if max_depth is not None and dist[u] == max_depth:
            continue
        for v in g.adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def dfs(g, s):
    # 4. iterative DFS with a stack, returns visit order, O(V + E)
    seen = {s}
    stack = [s]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in g.adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return order

def components(g):
    # 5. connected components via BFS, returns (label per node, count)
    comp = [-1] * g.n
    c = 0
    for s in range(g.n):
        if comp[s] == -1:
            for v in bfs(g, s):
                comp[v] = c
            c += 1
    return comp, c