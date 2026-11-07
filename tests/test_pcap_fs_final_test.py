import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcap_fs_final_test import (choose_columns, fit_setting, run_fs_final_test, select_settings, setting_tag,
                                compare_with_grid_test, training_rows)


def make_summary():
    rows = []
    for method, param in (("MI_top", 3.0), ("HeapBFS", 0.5)):
        for clf, f1 in (("RF", 0.30), ("XGB", 0.25), ("DT", 0.20), ("SVM", 0.10)):
            rows.append({"scaler": "Standard", "method": method, "param": param, "classifier": clf,
                         "f1_macro_mean": f1 + (0.01 if method == "HeapBFS" else 0), "accuracy_mean": 0.6})
    return pd.DataFrame(rows)


def make_masks():
    return pd.DataFrame([{"algorithm": "GA", "seed": 2, "scaler": "Normalizer", "classifier": "DT",
                          "features": "b;c", "cv_f1_macro": 0.26},
                         {"algorithm": "PSO", "seed": 1, "scaler": "Normalizer", "classifier": "DT",
                          "features": "a", "cv_f1_macro": 0.25}])


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = np.hstack([y[:, None] + rng.normal(scale=0.3, size=(n, 1)), rng.normal(size=(n, 3))])
    return X, y


def test_select_settings_merges_duplicates_and_adds_the_best_mask():
    settings = select_settings(make_summary(), make_masks())
    assert len(settings) == 4 + 4 + 1
    best_heap = next(s for s in settings if s["method"] == "HeapBFS" and s["classifier"] == "RF")
    assert best_heap["chosen_as"] == "best HeapBFS overall; best HeapBFS for RF"
    mask = settings[-1]
    assert mask["method"] == "Mask" and mask["features"] == "b;c" and mask["param"] == "GA seed 2"


def test_setting_tags():
    assert setting_tag({"scaler": "Normalizer", "method": "HeapBFS", "param": 0.84, "classifier": "DT"}) == \
        "Normalizer_HeapBFS_t0.84_DT"
    assert setting_tag({"scaler": "Standard", "method": "MI_top", "param": 22.0, "classifier": "XGB"}) == \
        "Standard_MI_top_k22_XGB"
    assert setting_tag({"scaler": "Normalizer", "method": "Mask", "param": "GA seed 2", "classifier": "DT"}) == \
        "Normalizer_Mask_GA_seed_2_DT"


def test_only_svm_is_subsampled():
    _, y = make_data()
    assert len(training_rows({"classifier": "SVM"}, y, 100, 42)) == 100
    assert len(training_rows({"classifier": "DT"}, y, 100, 42)) == 300


def test_columns_are_recomputed_once_per_scaler_and_row_count():
    X, y = make_data()
    cache = {}
    names = ["a", "b", "c", "d"]
    mi = choose_columns({"method": "MI_top", "param": 1.0, "scaler": "Standard"}, X, y, names, cache)
    heap = choose_columns({"method": "HeapBFS", "param": 1.0, "scaler": "Standard"}, X, y, names, cache)
    mask = choose_columns({"method": "Mask", "features": "d;b", "scaler": "Standard"}, X, y, names, cache)
    assert mi == [0] and heap == [0, 1, 2, 3] and mask == [3, 1] and len(cache) == 1


def test_fit_setting_predicts_with_the_chosen_columns():
    X, y = make_data()
    X[0, 1] = np.nan
    setting = {"scaler": "Standard", "method": "MI_top", "param": 2.0, "classifier": "DT"}
    fitted, n_train = fit_setting(setting, X, y, ["a", "b", "c", "d"], {})
    assert n_train == 300 and len(fitted.cols) == 2 and 0 in fitted.cols
    assert fitted.predict(X).shape == (300,)


def test_comparison_puts_both_studies_in_one_table(tmp_path):
    ours = pd.DataFrame([{"scaler": "Normalizer", "method": "HeapBFS", "param": 0.84, "classifier": "DT",
                          "n_features": 20, "chosen_as": "best HeapBFS overall", "cv_f1_macro": 0.25,
                          "test_f1_macro": 0.31, "test_balanced_accuracy": 0.3, "test_accuracy": 0.6}])
    pd.DataFrame([{"scaler": "Normalizer", "reduction": "PCA", "k": 14, "classifier": "DT",
                   "chosen_as": "C3 (DT) by macro-F1", "cv_f1_macro": 0.27, "test_f1_macro": 0.30,
                   "test_balanced_accuracy": 0.3, "test_accuracy": 0.65}]).to_csv(tmp_path / "grid.csv", index=False)
    table = compare_with_grid_test(ours, tmp_path / "grid.csv")
    assert list(table.method) == ["HeapBFS", "PCA"] and table.n_features.tolist() == [20, 14]


def test_refuses_to_test_twice(tmp_path):
    (tmp_path / "test_results.csv").write_text("done", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already been tested"):
        run_fs_final_test(out_dir=tmp_path)
