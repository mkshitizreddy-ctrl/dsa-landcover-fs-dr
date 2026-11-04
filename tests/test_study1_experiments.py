import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from landsat_data import standardize
from run_complexity import TIMING_COLUMNS, growth_exponents, run_complexity
from run_pca_experiment import SETTINGS, make_synthetic_data, run_pca_experiment
from run_selection import compare_selection_methods


def test_standardize_uses_training_statistics_only():
    X_train = np.array([[1.0, 5.0], [3.0, 5.0]])
    X_test = np.array([[2.0, 7.0]])
    train, test = standardize(X_train, X_test)
    assert np.allclose(train.mean(axis=0), 0)
    assert np.allclose(test, [[0.0, 2.0]])


def test_selection_methods_agree_with_naive_at_every_threshold():
    X_train, y_train, _, _ = make_synthetic_data()
    table, top = compare_selection_methods(X_train, y_train)
    assert table["same_as_naive"].all()
    assert len(top) == 5


def test_pca_experiment_runs_every_setting_on_synthetic_data():
    results = run_pca_experiment(synthetic=True)
    assert list(results["setting"]) == [label for label, _, _ in SETTINGS]
    assert results.loc[0, "dims"] == 12
    assert results["accuracy"].between(0, 1).all()


def test_complexity_runs_on_small_sizes_without_fitting_exponents():
    results, exponents, _ = run_complexity(feature_counts=(20, 40), trials=1)
    assert list(results["d"]) == [20, 40]
    assert exponents == {}
    assert set(growth_exponents(results, min_d=20)) <= set(TIMING_COLUMNS)
