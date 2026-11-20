import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcap_sample_level import aggregate_by_hash, hash_folds


def packets():
    X = pd.DataFrame({"a": [1.0, 3.0, 5.0, 10.0, np.nan, 7.0], "b": [0.0, 0.0, 6.0, 2.0, 4.0, 1.0]})
    y = np.array([2, 2, 2, 0, 0, 1])
    groups = np.array(["h1", "h1", "h1", "h2", "h2", "h3"])
    return X, y, groups


def test_aggregate_gives_one_row_per_hash_with_every_statistic():
    table, labels = aggregate_by_hash(*packets())
    assert list(table.index) == ["h1", "h2", "h3"]
    assert list(table.columns) == ["a_mean", "a_std", "a_min", "a_max", "b_mean", "b_std", "b_min", "b_max",
                                   "packets"]
    assert table.loc["h1", ["a_mean", "a_min", "a_max", "packets"]].tolist() == [3.0, 1.0, 5.0, 3]
    assert table.loc["h1", "a_std"] == pytest.approx(pd.Series([1.0, 3.0, 5.0]).std())
    assert labels.tolist() == [2, 0, 1]


def test_aggregate_ignores_missing_values_and_gives_one_packet_zero_spread():
    table, _ = aggregate_by_hash(*packets())
    assert table.loc["h2", ["a_mean", "a_min", "a_max"]].tolist() == [10.0, 10.0, 10.0]
    assert table.loc["h3", ["a_std", "b_std"]].tolist() == [0.0, 0.0]


def test_aggregate_matches_a_direct_groupby():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x", "y", "z"])
    groups = rng.integers(0, 15, 200).astype(str)
    y = np.array([int(g) % 3 for g in groups])
    table, labels = aggregate_by_hash(X, y, groups)
    expected = X.assign(Hash=groups).groupby("Hash").max()
    assert np.allclose(table[["x_max", "y_max", "z_max"]].to_numpy(), expected.to_numpy())
    assert labels.tolist() == [int(g) % 3 for g in table.index]
    assert table.packets.sum() == 200


def test_aggregate_refuses_a_hash_with_two_families():
    X, y, groups = packets()
    y[1] = 0
    with pytest.raises(ValueError, match="more than one family"):
        aggregate_by_hash(X, y, groups)


def test_hash_folds_maps_each_hash_to_its_validation_fold():
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])), (np.array([0, 1, 5]), np.array([2, 3, 4]))]
    assert hash_folds(groups, folds) == {"a": 0, "b": 1, "c": 1}


def test_hash_folds_refuses_a_hash_in_two_folds():
    groups = np.array(["a", "a", "b"])
    folds = [(np.array([1, 2]), np.array([0])), (np.array([0, 2]), np.array([1]))]
    with pytest.raises(ValueError, match="more than one validation fold"):
        hash_folds(groups, folds)
