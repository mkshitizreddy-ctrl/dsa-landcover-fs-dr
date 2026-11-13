import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from landsat_explorer import evaluate, feature_names, score_table, select, threshold_sweep, validation_split


def make_data():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 3))
    X = np.column_stack([base[:, j % 3] + 0.1 * rng.normal(size=300) for j in range(6)])
    C = np.abs(np.corrcoef(X, rowvar=False))
    return X, C, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def test_feature_names_follow_pixels_and_bands():
    names = feature_names()
    assert len(names) == 36 and names[0] == "pixel 1, band 1" and names[35] == "pixel 9, band 4"


@pytest.mark.parametrize("method", ["heap_bfs", "per_component", "naive"])
def test_every_method_keeps_one_feature_per_correlated_group(method):
    _, C, scores = make_data()
    result = select(scores, C, 0.9, method)
    assert result["features"] == [0, 1, 2] and result["components"] == 3 and result["edges"] == 3


def test_unknown_method_is_refused():
    _, C, scores = make_data()
    with pytest.raises(ValueError, match="Unknown method"):
        select(scores, C, 0.9, "random")


def test_threshold_sweep_keeps_everything_at_one():
    _, C, scores = make_data()
    sweep = threshold_sweep(scores, C, [0.5, 1.0])
    assert sweep.features_kept.tolist() == [3, 6]


def test_validation_split_is_standardized_and_stratified():
    X_train, X_valid, y_train, y_valid = validation_split()
    assert len(X_train) + len(X_valid) == 4435 and abs(len(X_valid) - 887) <= 1
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-9)
    assert set(np.unique(y_train)) == set(np.unique(y_valid))


def test_evaluate_with_and_without_pca():
    X_train, X_valid, y_train, y_valid = validation_split()
    plain = evaluate(X_train[:600], y_train[:600], X_valid[:100], y_valid[:100], [0, 1, 2, 3], k=3)
    reduced = evaluate(X_train[:600], y_train[:600], X_valid[:100], y_valid[:100], list(range(36)), k=3,
                       pca_ratio=0.9)
    assert plain["dimensions"] == 4 and reduced["dimensions"] < 36
    assert 0 <= plain["accuracy"] <= 1 and reduced["query_seconds"] >= 0


def test_score_table_marks_kept_features():
    table = score_table([0.1, 0.3, 0.2], [1])
    assert table.feature.tolist()[0] == "pixel 1, band 2" and table.status.tolist() == ["kept", "removed", "removed"]
