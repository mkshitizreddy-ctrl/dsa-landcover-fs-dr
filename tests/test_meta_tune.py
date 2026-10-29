import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph import build_corr_graph
from meta_fs import ALGORITHMS
from meta_tune import PipelineObjective, decode_k, grid_search, run_tuning
from selection import select_features


def make_data(seed=0):
    """4 informative features, each with a near-duplicate, + 4 noise. D = 12."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 80)                             # Shape [240]
    base = y[:, None] * 1.5 + rng.normal(size=(240, 4))         # Shape [240, 4]
    dup = base + 0.2 * rng.normal(size=(240, 4))                # highly correlated copies
    noise = rng.normal(size=(240, 4))
    X = np.hstack([base, dup, noise])                           # Shape [240, 12]
    idx = rng.permutation(240)
    tr, va = idx[:180], idx[180:]
    return X[tr], y[tr], X[va], y[va]


def test_decode_k_is_exact_and_feasible():
    for m in range(1, 40):
        for k in range(1, m + 1):
            assert decode_k(k / m, m) == k
        assert decode_k(0.0, m) == 1 and decode_k(1.0, m) == m


def test_cached_subset_matches_direct_heap_bfs():
    obj = PipelineObjective(*make_data(), mode="threshold")
    for t in np.linspace(0.0, 1.0, 41):
        direct = tuple(sorted(select_features(obj.scores, build_corr_graph(obj.C, t))))
        assert obj.subset_for(t) == direct


def test_threshold_extremes():
    obj = PipelineObjective(*make_data(), mode="threshold")
    assert len(obj.subset_for(1.0)) == obj.D          # no edges: keep everything
    assert len(obj.subset_for(0.0)) < obj.D           # dense graph: few survive


def test_custom_pca_gives_same_fitness_as_svd():
    data = make_data()
    custom = grid_search(*data, mode="joint", pca_backend="custom")
    svd = grid_search(*data, mode="joint", pca_backend="svd")
    assert custom.best_fitness == pytest.approx(svd.best_fitness, abs=1e-12)
    assert custom.n_components == svd.n_components


@pytest.mark.parametrize("mode", ["threshold", "components", "joint"])
def test_no_metaheuristic_beats_the_grid_optimum(mode):
    data = make_data()
    optimum = grid_search(*data, mode=mode).best_fitness
    for name in sorted(ALGORITHMS):
        res = run_tuning(name, *data, mode=mode, population=10, seed=0, max_evaluations=40)
        assert res.best_fitness >= optimum - 1e-12
        assert 1 <= res.n_components <= len(res.selected_indices)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        PipelineObjective(*make_data(), mode="both")
