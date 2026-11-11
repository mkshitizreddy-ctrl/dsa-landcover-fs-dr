import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcap_class_weights import choose_configs, class_weights, config_features, summarize_weighting
from pcap_feature_selection import feature_statistics
from pcap_grid import evaluate


def test_no_weighting_returns_none():
    assert class_weights(np.array([0, 0, 1]), "none") is None


def test_balanced_weights_give_every_class_the_same_total():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = class_weights(y, "balanced")
    totals = [w[y == c].sum() for c in (0, 1, 2)]
    assert np.allclose(totals, totals[0]) and w.mean() == pytest.approx(1.0)


def test_sqrt_weights_are_milder_and_average_one():
    y = np.array([0] * 90 + [1] * 10)
    sqrt, full = class_weights(y, "sqrt"), class_weights(y, "balanced")
    assert sqrt.mean() == pytest.approx(1.0)
    assert sqrt[y == 1][0] / sqrt[y == 0][0] == pytest.approx(3.0)
    assert full[y == 1][0] / full[y == 0][0] == pytest.approx(9.0)


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        class_weights(np.array([0, 1]), "double")


def test_weighting_helps_a_rare_class_be_recalled():
    rng = np.random.default_rng(0)
    y = np.array([0] * 950 + [1] * 50)
    X = rng.normal(size=(1000, 2)) + np.where(y == 1, 0.8, 0.0)[:, None]
    plain = evaluate("SVM", X, y, X, y, [0, 1])
    weighted = evaluate("SVM", X, y, X, y, [0, 1], sample_weight=class_weights(y, "balanced"))
    assert weighted["f1_class1"] > plain["f1_class1"]


def test_choose_configs_takes_the_best_of_each_classifier_from_both_studies():
    grid = pd.DataFrame([{"scaler": "Normalizer", "reduction": "PCA", "k": k, "classifier": clf,
                          "f1_macro_mean": 0.2 + k / 100} for clf in ("RF", "XGB", "DT", "SVM") for k in (5, 9)])
    fs = pd.DataFrame([{"scaler": "Standard", "method": "HeapBFS", "param": t, "classifier": clf,
                        "f1_macro_mean": 0.2 + t / 10} for clf in ("RF", "XGB", "DT", "SVM") for t in (0.5, 0.8)])
    configs = choose_configs(grid, fs)
    assert len(configs) == 8
    assert {c["param"] for c in configs if c["family"] == "extraction"} == {9.0}
    assert {c["param"] for c in configs if c["family"] == "selection"} == {0.8}


def test_config_features_for_extraction_and_selection():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=200)
    X = np.hstack([y[:, None] + rng.normal(scale=0.3, size=(200, 1)), rng.normal(size=(200, 4))])
    stats = feature_statistics(X, y)
    pca = {"method": "PCA", "param": 3.0}
    mi = {"method": "MI_top", "param": 2.0}
    heap = {"method": "HeapBFS", "param": 1.0}
    assert config_features(pca, X, y, X[:10], stats)[0].shape == (200, 3)
    assert config_features(mi, X, y, X[:10], stats)[1].shape == (10, 2)
    assert config_features(heap, X, y, X[:10], stats)[0].shape == (200, 5)


def test_summary_averages_folds_per_scheme():
    rows = [{"family": "selection", "scaler": "Standard", "method": "HeapBFS", "param": 0.8, "classifier": "DT",
             "scheme": scheme, "fold": fold, "status": "success", "f1_macro": f1, "balanced_accuracy": f1,
             "accuracy": f1, "f1_class0": f1, "f1_class1": f1}
            for scheme, f1 in (("none", 0.2), ("balanced", 0.3)) for fold in (1, 2)]
    summary = summarize_weighting(pd.DataFrame(rows), ["Benign", "Gafgyt"])
    assert list(summary.scheme) == ["none", "balanced"] and list(summary.f1_Gafgyt) == [0.2, 0.3]
