class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)


def build_corr_graph(C, t):
    d = len(C)
    g = Graph(d)
    for i in range(d):
        for j in range(i + 1, d):
            if abs(C[i][j]) > t:
                g.add_edge(i, j)
    return g


def bfs(g, s, max_depth=None):
    dist = {s: 0}
    queue = [s]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if max_depth is not None and dist[u] == max_depth:
            continue
        for v in g.adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def dfs(g, s):
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
    comp = [-1] * g.n
    count = 0
    for start in range(g.n):
        if comp[start] == -1:
            for vertex in bfs(g, start):
                comp[vertex] = count
            count += 1
    return comp, count
