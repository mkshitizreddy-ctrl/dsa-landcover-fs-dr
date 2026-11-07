import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcap_feature_selection import THRESHOLDS
from pcap_fs_search import SelectionSpace, run_selection_search, search_odds
from pcap_grid import CLASSIFIERS, SCALERS
from pcap_mh_search import RANDOM, run_random


def make_table(n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for scaler in SCALERS:
        for clf in CLASSIFIERS:
            for k in range(1, n_features + 1):
                rows.append({"scaler": scaler, "method": "MI_top", "param": float(k), "classifier": clf,
                             "f1_macro_mean": rng.random() * 0.5})
            for t in THRESHOLDS:
                rows.append({"scaler": scaler, "method": "HeapBFS", "param": t, "classifier": clf,
                             "f1_macro_mean": rng.random() * 0.5})
    rows[0]["f1_macro_mean"] = 0.9
    return pd.DataFrame(rows)


def test_decode_covers_both_methods():
    space = SelectionSpace(make_table(), n_features=2)
    assert space.decode([0, 0, 0, 0]) == ("Standard", "MI_top", 1.0, "RF")
    assert space.decode([0, 0, 1, 0]) == ("Standard", "MI_top", 2.0, "RF")
    assert space.decode([0.99, 0.99, 0.5, 0.99]) == ("Normalizer", "HeapBFS", 0.5, "SVM")
    assert space.decode([0.5, 0.6, 1.0, 0.3]) == ("MinMax", "HeapBFS", 1.0, "XGB")


def test_best_setting_and_rank():
    space = SelectionSpace(make_table(), n_features=2)
    assert space.best_cfg == ("Standard", "MI_top", 1.0, "RF")
    assert space.rank(space.best_cfg) == 1
    assert space.n == 3 * 4 * (2 + 101)


def test_incomplete_table_is_refused():
    with pytest.raises(ValueError, match="expected"):
        SelectionSpace(make_table().iloc[:-1], n_features=2)


def test_random_search_finds_an_easy_best():
    space = SelectionSpace(make_table(), n_features=2)
    result = run_random(space, budget=500, seed=1)
    assert result["hit_best"] and result["best_rank"] == 1


def test_odds_count_tied_settings():
    table = make_table()
    table.loc[1, "f1_macro_mean"] = 0.9
    space = SelectionSpace(table, n_features=2)
    assert space.count_within(0.0) == 2
    odds = search_odds(space, draws=(1,))
    assert odds["finds_best_%"].iloc[0] == pytest.approx(100 * 2 / space.n)


def test_run_uses_selection_column_names(tmp_path):
    make_table().to_csv(tmp_path / "summary.csv", index=False)
    runs, summary, _ = run_selection_search(tmp_path / "summary.csv", algorithms=(RANDOM,), budgets=(20,),
                                            n_seeds=2, n_features=2, save=False, show=lambda message: None)
    assert {"method", "param"} <= set(runs.columns) and "reduction" not in runs.columns
    assert len(runs) == 2 and list(summary.algorithm) == [RANDOM]
