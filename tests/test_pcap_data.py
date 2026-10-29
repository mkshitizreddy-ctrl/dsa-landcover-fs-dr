import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcap_data import load_pcap, load_splits, make_splits, save_splits


def write_synthetic_csv(path, seed=0):
    """Same schema as pcap.csv: 6 kept classes x 12 hashes, plus Agent/Tsunami."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, family in enumerate(["Benign", "DarkNexus", "Gafgyt", "Generic", "Mirai", "Unknown"]):
        for h in range(12):
            for _ in range(int(rng.integers(3, 15))):          # uneven rows per hash
                rows.append({"f1": c + rng.normal(), "f2": rng.normal(), "const": 1.0,
                             "Rate": rng.exponential(), "Hash": f"{family}_{h}",
                             "MalwareFamily": family, "Arch": "arm"})
    for family, n_hashes in [("Agent", 1), ("Tsunami", 3)]:
        for h in range(n_hashes):
            rows.append({"f1": 0.0, "f2": 0.0, "const": 1.0, "Rate": 1.0,
                         "Hash": f"{family}_{h}", "MalwareFamily": family, "Arch": "arm"})
    df = pd.DataFrame(rows)
    df.loc[0, "Rate"] = np.inf
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "synthetic.csv"
    write_synthetic_csv(path)
    return load_pcap(path)


def test_load_drops_classes_and_non_features(data):
    assert data.dropped_rows == 4
    assert data.class_names == sorted(["Benign", "DarkNexus", "Gafgyt", "Generic", "Mirai", "Unknown"])
    assert list(data.X.columns) == ["f1", "f2", "const", "Rate"]
    assert data.X["Rate"].isna().sum() == 1 and not np.isinf(data.X.to_numpy()).any()


def test_holdout_is_grouped_and_has_every_class(data):
    s = make_splits(data)
    assert not set(data.groups[s.dev_idx]) & set(data.groups[s.test_idx])
    assert len(np.intersect1d(s.dev_idx, s.test_idx)) == 0
    assert len(s.dev_idx) + len(s.test_idx) == len(data.y)
    for part in (s.dev_idx, s.test_idx):
        assert set(data.y[part]) == set(range(6))


def test_sample_respects_cap_and_stays_in_dev(data):
    s = make_splits(data, rows_per_hash=5)
    assert set(s.sample_idx) <= set(s.dev_idx)
    assert pd.Series(data.groups[s.sample_idx]).value_counts().max() <= 5


def test_folds_partition_the_sample_without_hash_leaks(data):
    s = make_splits(data, cv_folds=4)
    valid_all = np.concatenate([va for _, va in s.folds])
    assert sorted(valid_all.tolist()) == sorted(s.sample_idx.tolist())   # each row validated once
    for tr, va in s.folds:
        assert not set(data.groups[tr]) & set(data.groups[va])
        assert set(data.y[tr]) == set(data.y[va]) == set(range(6))


def test_constant_columns_dropped(data):
    s = make_splits(data)
    assert "const" not in s.features and "f1" in s.features


def test_same_seed_same_splits(data):
    a, b = make_splits(data, seed=7), make_splits(data, seed=7)
    assert (a.test_idx == b.test_idx).all() and (a.sample_idx == b.sample_idx).all()


def test_save_load_roundtrip_and_row_check(data, tmp_path):
    s = make_splits(data)
    path = tmp_path / "splits.npz"
    save_splits(s, path)
    r = load_splits(path, n_rows=len(data.y))
    assert (r.test_idx == s.test_idx).all() and r.features == s.features
    assert all((a == b).all() for f1, f2 in zip(r.folds, s.folds) for a, b in zip(f1, f2))
    with pytest.raises(ValueError, match="regenerate"):
        load_splits(path, n_rows=len(data.y) + 1)
