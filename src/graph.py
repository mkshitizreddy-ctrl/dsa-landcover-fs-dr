
class Graph:
    """Undirected graph represented by an adjacency list."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        """Add an undirected edge between vertices u and v."""
        self.adj[u].append(v)
        self.adj[v].append(u)


def build_corr_graph(C, t):
    """
    Build an undirected feature-correlation graph.

    Add an edge between features i and j when abs(C[i][j]) > t.
    C may contain signed correlations or absolute correlations.

    Time complexity: O(d^2), where d is the number of features.
    """
    d = len(C)
    g = Graph(d)

    for i in range(d):
        for j in range(i + 1, d):
            if abs(C[i][j]) > t:
                g.add_edge(i, j)

    return g


def bfs(g, s, max_depth=None):
    """
    Breadth-first search returning {node: distance}.

    Time complexity: O(V + E) for a full traversal.
    """
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
    """
    Iterative depth-first search returning visit order.

    Time complexity: O(V + E) for a full traversal.
    """
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
    """
    Find connected components using BFS.

    Returns (component_label_per_vertex, component_count).
    """
    comp = [-1] * g.n
    count = 0

    for start in range(g.n):
        if comp[start] == -1:
            for vertex in bfs(g, start):
                comp[vertex] = count
            count += 1

    return comp, count
