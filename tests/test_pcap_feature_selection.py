import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcap_feature_selection import (best_per_method, compare_with_grid, feature_statistics, heap_bfs_subsets,
                                    mi_top_subsets, prepare_fold, score_subsets, summarize)


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    signal = y[:, None] + rng.normal(scale=0.5, size=(n, 1))
    X = np.hstack([signal, signal * 2 + rng.normal(scale=0.01, size=(n, 1)), rng.normal(size=(n, 3))])
    return X, y


def test_prepare_fold_imputes_and_scales_on_training_rows():
    X, _ = make_data()
    X[0, 2] = np.nan
    Z_train, Z_valid = prepare_fold(X, np.arange(200), np.arange(200, 300))
    assert not np.isnan(Z_train).any() and not np.isnan(Z_valid).any()
    np.testing.assert_allclose(Z_train.mean(axis=0), 0, atol=1e-9)
    N_train, _ = prepare_fold(X, np.arange(200), np.arange(200, 300), "Normalizer")
    np.testing.assert_allclose(np.linalg.norm(N_train, axis=1), 1, atol=1e-9)


def test_mi_top_keeps_the_highest_scores():
    subsets = mi_top_subsets([0.1, 0.9, 0.5, 0.3], [1, 2, 4])
    assert subsets == {1: (1,), 2: (1, 2), 4: (0, 1, 2, 3)}


def test_heap_bfs_drops_one_of_two_correlated_features():
    X, y = make_data()
    scores, C = feature_statistics(X, y)
    subsets = heap_bfs_subsets(scores, C, [0.9, 1.0])
    assert subsets[1.0] == (0, 1, 2, 3, 4)
    assert len(subsets[0.9]) == 4 and len({0, 1} & set(subsets[0.9])) == 1


def test_constant_column_gives_zero_correlation():
    X, y = make_data()
    X[:, 4] = 1.0
    _, C = feature_statistics(X, y)
    assert np.isfinite(C).all() and C[4, :4].max() == 0


def test_each_distinct_subset_is_scored_once():
    X, y = make_data()
    scored = score_subsets("DT", [(0, 1), (0, 1), (2,)], X[:200], y[:200], X[200:], y[200:], [0, 1, 2], 42, 1)
    assert set(scored) == {(0, 1), (2,)}
    assert all(r["status"] == "success" for r in scored.values())


def test_summary_and_comparison_tables(tmp_path):
    rows = []
    for fold in (1, 2):
        for method, param, k, f1 in (("MI_top", 2.0, 2, 0.6), ("HeapBFS", 0.9, 3 + fold, 0.7)):
            rows.append({"scaler": "Standard", "method": method, "param": param, "classifier": "DT", "fold": fold,
                         "k": k, "f1_macro": f1, "balanced_accuracy": f1, "accuracy": f1, "status": "success"})
    pd.DataFrame(rows).to_csv(tmp_path / "fold.csv", index=False)
    summary = summarize(tmp_path / "fold.csv", 2, tmp_path / "summary.csv")
    assert list(summary.method) == ["HeapBFS", "MI_top"]
    assert summary.iloc[0].k_min == 4 and summary.iloc[0].k_max == 5

    grid = pd.DataFrame([{"scaler": "Standard", "reduction": "PCA", "k": 5, "classifier": "DT",
                          "f1_macro_mean": 0.65, "accuracy_mean": 0.8},
                         {"scaler": "MinMax", "reduction": "PCA", "k": 9, "classifier": "DT",
                          "f1_macro_mean": 0.99, "accuracy_mean": 0.99}])
    grid.to_csv(tmp_path / "grid.csv", index=False)
    table = compare_with_grid(summary, tmp_path / "grid.csv")
    assert list(table.method) == ["MI_top", "HeapBFS", "PCA"]
    assert table.set_index("method").loc["PCA", "cv_f1_macro"] == 0.65
    assert table.set_index("method").loc["HeapBFS", "setting"] == "t = 0.90"
    assert len(table) == 3 and set(table.scaler) == {"Standard"}
    assert best_per_method(table).iloc[0].method == "HeapBFS"
