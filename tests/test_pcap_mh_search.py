import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meta_fs import ALGORITHMS
from pcap_grid import CLASSIFIERS, REDUCTIONS, SCALERS, k_values
from pcap_mh_search import GridSpace, Tracker, p_random_hit, run_algorithm, run_random


def make_table(seed=0):
    """All 924 configurations with random scores and one clear best."""
    rng = np.random.default_rng(seed)
    rows = [dict(scaler=s, reduction=r, k=k, classifier=c, f1_macro_mean=rng.uniform(0.1, 0.25))
            for s, r, c in product(SCALERS, REDUCTIONS, CLASSIFIERS) for k in k_values(r, 36, 6)]
    table = pd.DataFrame(rows)
    table.loc[100, "f1_macro_mean"] = 0.30
    table.loc[101, "f1_macro_mean"] = 0.295         # within 0.01 of the best
    return table


@pytest.fixture
def space():
    return GridSpace(make_table())


def test_decoding_reaches_every_configuration(space):
    found = set()
    for i, j, c in product(range(3), range(3), range(4)):
        top = space.k_top[REDUCTIONS[j]]
        for k in range(1, top + 1):
            pos = [(i + 0.5) / 3, (j + 0.5) / 3, (k - 0.5) / top, (c + 0.5) / 4]
            found.add(space.decode(pos))
    assert found == set(space.score) and len(found) == 924


def test_decoding_handles_bounds(space):
    assert space.decode([0, 0, 0, 0]) == ("Standard", "PCA", 1, "RF")
    assert space.decode([1, 1, 1, 1]) == ("Normalizer", "LDA", 5, "SVM")


def test_incomplete_table_rejected():
    with pytest.raises(ValueError, match="expected 924"):
        GridSpace(make_table().iloc[:-1])


def test_ranks_and_near_count(space):
    assert space.best == pytest.approx(0.30) and space.rank(space.best_cfg) == 1
    assert space.count_within(0.01) == 2
    worst = min(space.score, key=space.score.get)
    assert space.rank(worst) == 924


def test_tracker_counts_unique_and_hits(space):
    t = Tracker(space, near=0.01)
    best = space.best_cfg
    other = next(cfg for cfg in space.score if space.rank(cfg) > 2)
    to_pos = {cfg: None for cfg in (best, other)}
    for i, j, c in product(range(3), range(3), range(4)):
        top = space.k_top[REDUCTIONS[j]]
        for k in range(1, top + 1):
            pos = [(i + 0.5) / 3, (j + 0.5) / 3, (k - 0.5) / top, (c + 0.5) / 4]
            if space.decode(pos) in to_pos:
                to_pos[space.decode(pos)] = pos
    t(to_pos[other]); t(to_pos[other]); t(to_pos[best])
    assert t.calls == 3 and len(t.seen) == 2
    assert t.unique_to_best == 2 and t.unique_to_near == 2 and t.best_cfg == best
    assert t(to_pos[best]) == pytest.approx(1 - 0.30)


def test_random_search_uses_exact_budget(space):
    r = run_random(space, 150, seed=3)
    assert r["calls"] == 150 and r["unique_configs"] <= 150 and r["best_rank"] >= 1
    assert r["best_f1"] == pytest.approx(space.score[(r["scaler"], r["reduction"], r["k"], r["classifier"])])


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_runs_on_the_grid(space, name):
    r = run_algorithm(name, space, budget=60, population=10, seed=1)
    assert 60 <= r["calls"] < 60 + 3 * 10
    assert 1 <= r["unique_configs"] <= r["calls"]
    assert r["best_f1"] <= space.best + 1e-12
    assert r["best_at_10"] <= r["best_at_50"] <= r["best_at_400"]


def test_same_seed_same_result(space):
    a = run_algorithm("GWO", space, 80, 10, seed=5)
    b = run_algorithm("GWO", space, 80, 10, seed=5)
    assert (a["best_f1"], a["unique_configs"]) == (b["best_f1"], b["unique_configs"])


def test_random_hit_probability():
    assert p_random_hit(924, 1, 924) == pytest.approx(1.0)
    assert p_random_hit(924, 1, 92) == pytest.approx(92 / 924)
