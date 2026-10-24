
import sys
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

from graph import build_corr_graph, bfs, dfs, components
from unionfind import UnionFind


def make_test_data():
    rng = np.random.default_rng(42)
    n = 300
    base = rng.normal(size=(n, 6))
    X = np.empty((n, 12))

    for i in range(6):
        X[:, 2 * i] = base[:, i]
        X[:, 2 * i + 1] = (
            base[:, i] + 0.03 * rng.normal(size=n)
        )

    return X


def make_graph():
    X = make_test_data()
    correlation = np.corrcoef(X, rowvar=False)
    graph = build_corr_graph(correlation, 0.9)
    return graph, correlation


def test_graph_edges_follow_absolute_correlation_threshold():
    graph, correlation = make_graph()

    assert graph.n == correlation.shape[0]

    for i in range(graph.n):
        for j in graph.adj[i]:
            assert i in graph.adj[j]
            assert abs(correlation[i, j]) > 0.9


def test_graph_includes_strong_negative_correlations():
    correlation = np.array([
        [1.00, -0.96,  0.89,  0.00],
        [-0.96, 1.00, -0.20,  0.00],
        [0.89, -0.20,  1.00, -0.95],
        [0.00,  0.00, -0.95,  1.00],
    ])

    graph = build_corr_graph(correlation, 0.9)

    assert 1 in graph.adj[0]
    assert 0 in graph.adj[1]
    assert 3 in graph.adj[2]
    assert 2 in graph.adj[3]

    # Correlations below the threshold must not create edges.
    assert 2 not in graph.adj[0]
    assert 0 not in graph.adj[2]
    assert 3 not in graph.adj[0]


def test_components_match_scipy_and_union_find():
    graph, _ = make_graph()
    n = graph.n

    rows = []
    cols = []

    for i in range(n):
        for j in graph.adj[i]:
            rows.append(i)
            cols.append(j)

    adjacency = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n),
    )

    _, scipy_labels = connected_components(
        adjacency, directed=False
    )

    scipy_groups = {
        frozenset(np.flatnonzero(scipy_labels == label))
        for label in np.unique(scipy_labels)
    }

    labels, count = components(graph)
    assert count == len(scipy_groups)

    our_groups = {
        frozenset(
            i for i, label in enumerate(labels) if label == c
        )
        for c in range(count)
    }
    assert our_groups == scipy_groups

    uf = UnionFind(n)

    for i in range(n):
        for j in graph.adj[i]:
            uf.union(i, j)

    uf_groups = {}

    for i in range(n):
        root = uf.find(i)
        uf_groups.setdefault(root, set()).add(i)

    assert {
        frozenset(group) for group in uf_groups.values()
    } == scipy_groups


def test_bfs_and_dfs_reach_same_component():
    graph, _ = make_graph()

    for start in range(graph.n):
        bfs_nodes = set(bfs(graph, start))
        dfs_nodes = set(dfs(graph, start))
        assert bfs_nodes == dfs_nodes


def test_bfs_depth_one_visits_start_and_neighbors():
    graph, _ = make_graph()

    for start in range(graph.n):
        visited = set(bfs(graph, start, max_depth=1))
        expected = {start} | set(graph.adj[start])

        assert visited == expected
