
import sys
import random

import numpy as np
from sklearn.metrics import mutual_info_score

sys.path.insert(0, "src")

from mi import discretize, mutual_info, relevance_scores
from sorting import merge_sort


def test_merge_sort_matches_python_sorted():
    rng = random.Random(42)
    values = [rng.randint(0, 20) for _ in range(300)]

    assert merge_sort(values) == sorted(values)
    assert merge_sort(values, reverse=True) == sorted(
        values, reverse=True
    )

    pairs = [
        (rng.randint(0, 5), i)
        for i in range(100)
    ]

    assert merge_sort(
        pairs,
        key=lambda p: p[0],
        reverse=True,
    ) == sorted(
        pairs,
        key=lambda p: p[0],
        reverse=True,
    )


def test_mutual_info_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))

    y = (
        X[:, 0] + 0.5 * rng.normal(size=500) > 0
    ).astype(int)

    for j in range(6):
        xb = discretize(X[:, j], 10)

        assert abs(
            mutual_info(xb, y) - mutual_info_score(y, xb)
        ) < 1e-9


def test_informative_feature_ranks_first():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))

    y = (
        X[:, 0] + 0.5 * rng.normal(size=500) > 0
    ).astype(int)

    scores = relevance_scores(X, y)
    order = merge_sort(
        list(range(6)),
        key=lambda j: scores[j],
        reverse=True,
    )

    assert order[0] == 0


def test_values_on_a_bin_edge_go_to_the_upper_bin():
    assert discretize(np.array([0.0, 2.9999999999999996, 10.0])).tolist() == [0, 3, 9]


def test_scaling_does_not_change_mutual_information():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 11, size=(2000, 4)).astype(float)
    y = (counts[:, 0] > 5).astype(int)
    standard = (counts - counts.mean(axis=0)) / counts.std(axis=0)
    min_max = (counts - counts.min(axis=0)) / (counts.max(axis=0) - counts.min(axis=0))
    for a, b in ((counts, standard), (counts, min_max)):
        assert np.allclose(relevance_scores(a, y), relevance_scores(b, y), atol=1e-12)
