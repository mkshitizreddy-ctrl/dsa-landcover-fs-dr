import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pca import PCA


def make_data():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((600, 12)) @ rng.standard_normal((12, 12))
    A[:, 3] += 2 * A[:, 0]
    return A


def standardized(A):
    return (A - A.mean(0)) / A.std(0)


def test_eigenvalues_match_numpy():
    A = make_data()
    model = PCA(var_ratio=1.0, max_components=6).fit(A)
    expected = np.sort(np.linalg.eigvalsh(np.cov(standardized(A).T)))[::-1][:6]
    assert np.allclose(model.values, expected, rtol=1e-4)


def test_variance_and_components_match_sklearn():
    A = make_data()
    model = PCA(var_ratio=1.0, max_components=6).fit(A)
    reference = SklearnPCA(n_components=6).fit(standardized(A))
    assert np.allclose(model.values, reference.explained_variance_, rtol=1e-4)
    assert np.allclose(np.abs(model.components), np.abs(reference.components_), atol=1e-3)


def test_components_are_orthonormal():
    model = PCA(var_ratio=1.0, max_components=6).fit(make_data())
    assert np.allclose(model.components @ model.components.T, np.eye(6), atol=1e-4)


def test_stops_once_variance_ratio_is_reached():
    A = make_data()
    model = PCA(var_ratio=0.95).fit(A)
    assert model.explained_ratio.sum() >= 0.95
    assert model.explained_ratio[:-1].sum() < 0.95
    assert model.transform(A).shape == (600, len(model.values))
