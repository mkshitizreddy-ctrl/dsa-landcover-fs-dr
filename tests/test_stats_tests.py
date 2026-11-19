import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stats_tests import (average_ranks, friedman_test, landsat_friedman, landsat_nemenyi,
                         nemenyi_critical_difference, nemenyi_pairs, score_matrix)


def make_runs(scores):
    rows = []
    for block, values in enumerate(scores):
        for algorithm, value in zip("ABC", values):
            rows.append({"block": block, "algorithm": algorithm, "score": value})
    return pd.DataFrame(rows)


def test_score_matrix_has_one_row_per_block_and_one_column_per_algorithm():
    table = score_matrix(make_runs([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]), ["block"], "algorithm", "score")
    assert table.shape == (2, 3) and table.loc[0, "A"] == pytest.approx(0.9)


def test_score_matrix_refuses_a_missing_score():
    runs = make_runs([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]).iloc[:-1]
    with pytest.raises(ValueError, match="every block"):
        score_matrix(runs, ["block"], "algorithm", "score")


def test_average_ranks_give_rank_one_to_the_best_and_share_ties():
    table = score_matrix(make_runs([[0.9, 0.8, 0.8], [0.9, 0.7, 0.7]]), ["block"], "algorithm", "score")
    ranks = average_ranks(table)
    assert ranks["A"] == 1.0 and ranks["B"] == ranks["C"] == 2.5


def test_friedman_matches_scipy_and_finds_a_clear_difference():
    rng = np.random.default_rng(0)
    scores = [[0.9 + e, 0.8 + e, 0.7 + e] for e in rng.normal(0, 0.01, 10)]
    result = friedman_test(score_matrix(make_runs(scores), ["block"], "algorithm", "score"))
    assert result["n_blocks"] == 10 and result["n_algorithms"] == 3
    assert result["statistic"] == pytest.approx(20.0) and result["p_value"] < 0.001


def test_friedman_refuses_fewer_than_three_algorithms():
    with pytest.raises(ValueError, match="three algorithms"):
        friedman_test(pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}))


def test_landsat_results_give_one_row_per_task():
    table = landsat_friedman()
    assert list(table.task) == ["fs", "threshold", "components", "joint"]
    assert (table.n_blocks == 15).all() and (table.n_algorithms == 10).all()
    assert table.p_value.between(0, 1).all()


def test_critical_difference_uses_the_published_q_values():
    assert nemenyi_critical_difference(3, 1) / np.sqrt(2) == pytest.approx(2.343, abs=1e-3)
    assert nemenyi_critical_difference(10, 1) / np.sqrt(110 / 6) == pytest.approx(3.164, abs=1e-3)


def test_critical_difference_shrinks_with_more_blocks():
    assert nemenyi_critical_difference(10, 60) < nemenyi_critical_difference(10, 15)


def test_nemenyi_pairs_list_every_pair_once_with_the_better_algorithm_first():
    scores = [[0.9, 0.8, 0.7]] * 20
    pairs = nemenyi_pairs(score_matrix(make_runs(scores), ["block"], "algorithm", "score"))
    assert len(pairs) == 3 and (pairs.rank_gap > 0).all()
    assert pairs.iloc[0].better == "A" and pairs.iloc[0].worse == "B"


def test_nemenyi_finds_only_the_large_rank_gap_significant():
    scores = [[0.9, 0.8, 0.7]] * 20
    pairs = nemenyi_pairs(score_matrix(make_runs(scores), ["block"], "algorithm", "score"))
    significant = {(p.better, p.worse) for p in pairs.itertuples() if p.significant}
    assert significant == {("A", "B"), ("A", "C"), ("B", "C")}
    few = nemenyi_pairs(score_matrix(make_runs(scores[:3]), ["block"], "algorithm", "score"))
    assert set(few[few.significant].apply(lambda p: (p.better, p.worse), axis=1)) == {("A", "C")}


def test_landsat_nemenyi_compares_all_45_pairs():
    pairs = landsat_nemenyi("fs")
    assert len(pairs) == 45
    assert pairs.critical_difference.iloc[0] == pytest.approx(3.498, abs=1e-3)
