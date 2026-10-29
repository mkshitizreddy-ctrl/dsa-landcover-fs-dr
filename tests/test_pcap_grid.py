import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcap_grid import (REDUCTIONS, evaluate, fit_reducer, k_values, load_done, summarize,
                       transform)


def make_data(n=300, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)                                  # Shape [n]
    X = rng.normal(size=(n, d)) + y[:, None] * 0.8                  # Shape [n, d]
    return X, y


def test_grid_has_924_configurations():
    n = sum(len(k_values(r, 36, 6)) for r in REDUCTIONS) * 3 * 4
    assert n == 924
    assert k_values("LDA", 36, 6) == [1, 2, 3, 4, 5]
    assert k_values("PCA", 36, 6, smoke=True) == [2, 10, 36]
    assert k_values("LDA", 36, 6, smoke=True) == [2, 5]


@pytest.mark.parametrize("method", ["PCA", "LDA"])
def test_slicing_equals_fitting_with_k(method):
    X, y = make_data()
    top = 2 if method == "LDA" else 8
    Z_all = transform(fit_reducer(method, X, y, top), X)
    for k in range(1, top + 1):
        Z_k = transform(fit_reducer(method, X, y, k), X)
        np.testing.assert_allclose(Z_all[:, :k], Z_k, atol=1e-8)


def test_kernel_pca_slicing_matches_up_to_sign():
    X, y = make_data(n=200)
    Z_all = transform(fit_reducer("KernelPCA", X, y, 6, kpca_solver="dense"), X)
    for k in (1, 3, 6):
        Z_k = transform(fit_reducer("KernelPCA", X, y, k, kpca_solver="dense"), X)
        np.testing.assert_allclose(np.abs(Z_all[:, :k]), np.abs(Z_k), atol=1e-6)


def test_kernel_pca_respects_training_cap():
    X, y = make_data(n=300)
    model = fit_reducer("KernelPCA", X, y, 4, kpca_max_train=100)
    assert model.X_fit_.shape[0] == 100
    assert transform(model, X).shape == (300, 4)


def test_chunked_transform_matches_plain():
    X, y = make_data(n=250)
    model = fit_reducer("PCA", X, y, 5)
    np.testing.assert_allclose(transform(model, X, chunk=37), model.transform(X))


@pytest.mark.parametrize("clf", ["RF", "XGB", "DT", "SVM"])
def test_evaluate_scores_every_classifier(clf):
    X, y = make_data()
    out = evaluate(clf, X[:200], y[:200], X[200:], y[200:], [0, 1, 2])
    assert out["status"] == "success"
    assert 0 <= out["f1_macro"] <= 1 and {"f1_class0", "f1_class1", "f1_class2"} <= out.keys()


def test_evaluate_returns_failure_instead_of_raising():
    X, y = make_data()
    out = evaluate("NotAClassifier", X, y, X, y, [0, 1, 2])
    assert out["status"] == "failed" and "Unknown classifier" in out["error"]


def test_resume_and_summary(tmp_path):
    path = tmp_path / "fold_scores.csv"
    rows = []
    for fold in (1, 2):
        for k, f1 in ((1, 0.5), (2, 0.7)):
            rows.append(dict(scaler="Standard", reduction="PCA", k=k, classifier="RF", fold=fold,
                             f1_macro=f1 + 0.01 * fold, balanced_accuracy=f1, accuracy=f1,
                             f1_class0=f1, f1_class1=f1, fit_seconds=1.0, status="success"))
    rows.append(dict(scaler="Standard", reduction="PCA", k=3, classifier="RF", fold=1,
                     status="failed"))
    pd.DataFrame(rows).to_csv(path, index=False)
    done = load_done(path)
    assert ("Standard", "PCA", 2, "RF", 2) in done and ("Standard", "PCA", 3, "RF", 1) not in done
    s = summarize(path, n_folds=2, class_names=["A", "B"], out_csv=tmp_path / "s.csv")
    assert list(s.k) == [2, 1]                                       # ranked by macro-F1
    assert s.iloc[0].f1_macro_mean == pytest.approx(0.715) and "f1_A_mean" in s.columns
