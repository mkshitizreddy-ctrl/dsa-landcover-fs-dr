import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcap_feature_selection import THRESHOLDS
from pcap_fs_search import (MaskObjective, SelectionSpace, mask_row, run_selection_search, search_mask,
                             search_odds, summarize_masks)
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


def make_folds(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = np.hstack([y[:, None] + rng.normal(scale=0.2, size=(n, 1)), rng.normal(size=(n, 3))])
    half = n // 2
    return [(X[:half], y[:half], X[half:], y[half:]), (X[half:], y[half:], X[:half], y[:half])]


def test_mask_objective_caches_and_scores():
    objective = MaskObjective(make_folds(), [0, 1, 2])
    first = objective([0.9, 0.1, 0.1, 0.1])
    second = objective([0.8, 0.2, 0.3, 0.4])
    assert first == second and objective.calls == 2 and len(objective.cache) == 1
    assert objective.best_cols == (0,) and objective.best_f1 > 0.8
    assert first == pytest.approx(0.9 * (1 - objective.best_f1) + 0.1 * 1 / 4)


def test_empty_mask_keeps_the_largest_position():
    objective = MaskObjective(make_folds(), [0, 1, 2])
    objective([0.1, 0.4, 0.2, 0.3])
    assert list(objective.cache) == [(1,)]


@pytest.mark.parametrize("name", [RANDOM, "PSO"])
def test_search_mask_respects_the_budget(name):
    objective, _ = search_mask(name, make_folds(), [0, 1, 2], budget=20, population=10, seed=1)
    assert len(objective.cache) <= 20 and objective.best_cols is not None
    assert objective.fold_data is None and len(objective.curve) == len(objective.cache)


def test_mask_rows_and_summary():
    objective, seconds = search_mask(RANDOM, make_folds(), [0, 1, 2], budget=15, population=10, seed=2)
    settings = {"run_id": "r", "budget": 15, "population": 10, "scaler": "Standard", "classifier": "DT",
                "alpha": 0.9}
    row = mask_row(RANDOM, 2, objective, seconds, ["a", "b", "c", "d"], settings)
    assert row["n_features"] == len(row["features"].split(";")) and row["family"] == "baseline"
    summary = summarize_masks([row, {**row, "seed": 3, "cv_f1_macro": 0.0}])
    assert summary.iloc[0].best_cv_f1_macro == row["cv_f1_macro"]
