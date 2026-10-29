import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcap_final_test import fit_config, predict_chunked, select_configs


def make_grid():
    rows = []
    rng = np.random.default_rng(0)
    for s, r, k, c in product(["Standard", "MinMax"], ["PCA", "LDA"], [1, 2], ["RF", "XGB", "DT", "SVM"]):
        rows.append(dict(scaler=s, reduction=r, k=k, classifier=c, accuracy_mean=rng.uniform(0.5, 0.6),
                         f1_macro_mean=rng.uniform(0.2, 0.25), balanced_accuracy_mean=0.3))
    g = pd.DataFrame(rows)
    g.loc[0, "accuracy_mean"] = 0.9        # Standard PCA 1 RF: best by accuracy
    g.loc[2, "f1_macro_mean"] = 0.5        # Standard PCA 1 DT: best by macro-F1
    return g


def test_select_configs_covers_both_rankings_and_every_classifier():
    picks = select_configs(make_grid())
    keys = {(p["scaler"], p["reduction"], p["k"], p["classifier"]) for p in picks}
    assert len(keys) == len(picks)                                   # no duplicates
    by_reason = {p["chosen_as"]: p for p in picks}
    acc_best = next(p for p in picks if "best overall by accuracy" in p["chosen_as"])
    f1_best = next(p for p in picks if "best overall by macro-F1" in p["chosen_as"])
    assert (acc_best["classifier"], acc_best["cv_accuracy"]) == ("RF", 0.9)
    assert (f1_best["classifier"], f1_best["cv_f1_macro"]) == ("DT", 0.5)
    reasons = " ".join(by_reason)
    for tag in ("C1 (RF)", "C2 (XGB)", "C3 (DT)", "C4 (SVM)"):
        assert f"{tag} by accuracy" in reasons and f"{tag} by macro-F1" in reasons


def make_data(n=400, d=6, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = rng.normal(size=(n, d)) + y[:, None]
    X[0, 0] = np.nan                                                 # imputation path
    return X, y


@pytest.mark.parametrize("reduction,k", [("PCA", 3), ("KernelPCA", 3), ("LDA", 2)])
@pytest.mark.parametrize("classifier", ["RF", "XGB", "DT", "SVM"])
def test_fit_config_runs_every_combination(reduction, k, classifier):
    X, y = make_data()
    cfg = dict(scaler="Normalizer", reduction=reduction, k=k, classifier=classifier)
    pipe, n_train = fit_config(cfg, X[:300], y[:300], kpca_max_train=150, svm_max_rows=200)
    pred = predict_chunked(pipe, X[300:], chunk=33)
    assert pred.shape == (100,) and set(pred) <= {0, 1, 2}
    assert n_train == (200 if classifier == "SVM" else 300)


def test_chunked_prediction_matches_plain():
    X, y = make_data()
    pipe, _ = fit_config(dict(scaler="Standard", reduction="PCA", k=4, classifier="DT"), X[:300], y[:300])
    assert (predict_chunked(pipe, X[300:], chunk=7) == pipe.predict(X[300:])).all()
