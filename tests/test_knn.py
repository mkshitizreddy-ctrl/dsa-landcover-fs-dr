import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kdtree import KDTree
from knn import KNN


def make_data(seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 5)) + np.repeat(np.arange(3), 200)[:, None]
    y = np.repeat(np.arange(3), 200)
    Q = rng.normal(size=(100, 5)) + rng.integers(0, 3, size=(100, 1))
    return X, y, Q


def test_kdtree_distances_match_brute_force():
    X, _, Q = make_data()
    tree = KDTree(X)
    for q in Q:
        got = [d for d, _ in tree.query(q, 5)]
        expected = np.sort(((X - q) ** 2).sum(axis=1))[:5]
        assert np.allclose(got, expected)


def test_knn_predictions_match_sklearn():
    X, y, Q = make_data()
    mine = KNN(5).fit(X, y).predict(Q)
    reference = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X, y).predict(Q)
    assert (mine == reference).all()


def test_knn_tie_goes_to_smallest_label_for_numbers_and_text():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    for labels in (np.array([2, 2, 1, 1]), np.array(["b", "b", "a", "a"])):
        prediction = KNN(4).fit(X, labels).predict(np.array([[5.5]]))
        assert prediction[0] == min(labels)
