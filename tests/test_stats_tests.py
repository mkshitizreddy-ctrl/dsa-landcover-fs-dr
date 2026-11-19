import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stats_tests import average_ranks, friedman_test, landsat_friedman, score_matrix


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
