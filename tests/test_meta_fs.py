import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meta_fs import ALGORITHMS, FSObjective, decode_mask, run_feature_selection


def make_data(seed=0):
    """3 informative features + 7 noise features, 3 classes."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 80)                           # Shape [240]
    informative = y[:, None] * 2.0 + rng.normal(size=(240, 3))  # Shape [240, 3]
    noise = rng.normal(size=(240, 7))                          # Shape [240, 7]
    X = np.hstack([informative, noise])                        # Shape [240, 10]
    idx = rng.permutation(240)
    tr, va = idx[:180], idx[180:]
    return X[tr], y[tr], X[va], y[va]


def test_decode_mask_threshold_and_never_empty():
    assert decode_mask([0.1, 0.6, 0.5]).tolist() == [False, True, True]
    assert decode_mask([0.1, 0.3, 0.2]).tolist() == [False, True, False]


def test_fitness_matches_formula():
    X_tr, y_tr, X_va, y_va = make_data()
    obj = FSObjective(X_tr, y_tr, X_va, y_va, alpha=0.9)
    position = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    mask = position >= 0.5
    pred = KNeighborsClassifier(5).fit(X_tr[:, mask], y_tr).predict(X_va[:, mask])
    expected = 0.9 * (1 - f1_score(y_va, pred, average="macro")) + 0.1 * 3 / 10
    assert obj(position) == pytest.approx(expected)


def test_cache_counts_calls_but_fits_once_per_mask():
    X_tr, y_tr, X_va, y_va = make_data()
    obj = FSObjective(X_tr, y_tr, X_va, y_va)
    a = obj(np.array([0.9, 0.8, 0.1] + [0.0] * 7))
    b = obj(np.array([0.7, 0.6, 0.4] + [0.2] * 7))  # same decoded mask
    assert a == b and obj.calls == 2 and len(obj.cache) == 1


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_runs_and_reports_consistently(name):
    X_tr, y_tr, X_va, y_va = make_data()
    res = run_feature_selection(name, X_tr, y_tr, X_va, y_va,
                                epochs=3, population=10, seed=1)
    assert res.mask.dtype == bool and res.mask.shape == (10,)
    assert 1 <= res.n_selected <= 10
    assert res.failures == 0
    assert len(res.convergence) == 3
    assert all(b <= a + 1e-12 for a, b in zip(res.convergence, res.convergence[1:]))
    # Our tracked best must agree with Mealpy's own global best.
    assert res.best_fitness == pytest.approx(res.convergence[-1])


def test_population_below_minimum_is_rejected_clearly():
    X_tr, y_tr, X_va, y_va = make_data()
    with pytest.raises(ValueError, match="population must be >= 10"):
        run_feature_selection("GA", X_tr, y_tr, X_va, y_va, epochs=2, population=8)


def test_same_seed_gives_same_mask():
    X_tr, y_tr, X_va, y_va = make_data()
    r1 = run_feature_selection("GWO", X_tr, y_tr, X_va, y_va, epochs=3, population=10, seed=7)
    r2 = run_feature_selection("GWO", X_tr, y_tr, X_va, y_va, epochs=3, population=10, seed=7)
    assert (r1.mask == r2.mask).all() and r1.best_fitness == r2.best_fitness


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_evaluation_budget_is_equalised_across_algorithms(name):
    X_tr, y_tr, X_va, y_va = make_data()
    budget, pop = 60, 10
    res = run_feature_selection(name, X_tr, y_tr, X_va, y_va,
                                population=pop, seed=3, max_evaluations=budget)
    # Mealpy stops at the end of the epoch that crosses the budget.
    assert budget <= res.calls < budget + 3 * pop
