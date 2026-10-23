
import sys

import numpy as np

sys.path.insert(0, "src")

from graph import build_corr_graph
from selection import (
    select_features,
    select_naive,
    select_per_component,
)


def make_test_data():
    rng = np.random.default_rng(42)
    n = 500

    # Six independent signals, each represented by a highly
    # correlated pair of features.
    base = rng.normal(size=(n, 6))
    X = np.empty((n, 12))

    for i in range(6):
        X[:, 2 * i] = base[:, i]
        X[:, 2 * i + 1] = (
            base[:, i] + 0.03 * rng.normal(size=n)
        )

    correlation = np.corrcoef(X, rowvar=False)
    return X, correlation


def test_heap_bfs_selection_matches_expected_greedy_selection():
    _, correlation = make_test_data()
    threshold = 0.9
    graph = build_corr_graph(correlation, threshold)

    scores = np.arange(graph.n, dtype=float)

    selected = select_features(scores, graph)

    # Each correlated pair is connected. The higher-scoring
    # member of each pair should be selected.
    expected = [1, 3, 5, 7, 9, 11]

    assert set(selected) == set(expected)
    assert len(selected) == len(expected)


def test_selected_features_have_no_direct_graph_edges():
    _, correlation = make_test_data()
    graph = build_corr_graph(correlation, 0.9)

    rng = np.random.default_rng(7)
    scores = rng.random(graph.n)

    selected = select_features(scores, graph)

    for i, feature in enumerate(selected):
        for other in selected[i + 1:]:
            assert other not in graph.adj[feature]


def test_naive_selection_matches_greedy_correlation_rule():
    _, correlation = make_test_data()
    threshold = 0.9

    scores = np.arange(correlation.shape[0], dtype=float)
    selected = select_naive(scores, correlation, threshold)

    # With the generated data, each correlated pair retains
    # its higher-scoring feature.
    expected = [11, 9, 7, 5, 3, 1]

    assert selected == expected


def test_per_component_selects_one_best_feature_per_component():
    _, correlation = make_test_data()
    graph = build_corr_graph(correlation, 0.9)

    scores = np.arange(graph.n, dtype=float)
    selected = select_per_component(scores, graph)

    expected = [1, 3, 5, 7, 9, 11]

    assert selected == expected
    assert len(selected) == len(set(selected))
