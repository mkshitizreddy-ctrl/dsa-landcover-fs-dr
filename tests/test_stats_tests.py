import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stats_tests import (average_ranks, cd_groups, compare_with_baseline, fisher_against_baseline, pcap_hit_rates, friedman_test, holm_adjust, kruskal_test,
                         landsat_friedman, landsat_nemenyi, nemenyi_critical_difference, nemenyi_pairs,
                         landsat_rank_table, pcap_against_random, pcap_kruskal, run_statistical_tests,
                         save_cd_diagrams, score_matrix)


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


def test_holm_adjustment_matches_a_hand_worked_example():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_kruskal_finds_a_clear_difference_and_refuses_two_groups():
    rng = np.random.default_rng(1)
    groups = {"A": rng.normal(0.0, 1, 20), "B": rng.normal(0.0, 1, 20), "C": rng.normal(3.0, 1, 20)}
    assert kruskal_test(groups)["p_value"] < 0.001
    with pytest.raises(ValueError, match="three algorithms"):
        kruskal_test({"A": groups["A"], "B": groups["B"]})


def test_baseline_comparison_flags_only_the_shifted_algorithm():
    rng = np.random.default_rng(2)
    groups = {"RANDOM": rng.normal(0, 1, 20), "SAME": rng.normal(0, 1, 20), "BETTER": rng.normal(3, 1, 20)}
    table = compare_with_baseline(groups)
    assert list(table.algorithm) == ["BETTER", "SAME"]
    assert list(table.significant) == [True, False]
    assert (table.p_holm >= table.p_value).all()


def test_pcap_kruskal_gives_one_row_per_budget_for_both_search_spaces():
    for space in ("grid", "selection"):
        table = pcap_kruskal(space)
        assert list(table.budget) == [100, 200, 400]
        assert (table.n_algorithms == 11).all() and (table.runs_per_algorithm == 20).all()


def test_pcap_against_random_compares_the_ten_metaheuristics():
    table = pcap_against_random("grid", 200)
    assert len(table) == 10 and "RANDOM" not in set(table.algorithm)
    with pytest.raises(ValueError, match="Unknown search space"):
        pcap_kruskal("other")


def test_landsat_rank_table_has_ten_algorithms_per_task_and_ranks_averaging_five_and_a_half():
    table = landsat_rank_table()
    assert len(table) == 40
    assert table.groupby("task").average_rank.mean().tolist() == pytest.approx([5.5] * 4)


def test_run_statistical_tests_saves_every_table(tmp_path):
    messages = []
    tables = run_statistical_tests(out_dir=tmp_path, show=messages.append)
    expected = {"landsat_friedman", "landsat_ranks", "landsat_nemenyi", "pcap_kruskal", "pcap_vs_random",
                "pcap_hit_rates"}
    assert set(tables) == expected
    assert {p.stem for p in tmp_path.glob("*.csv")} == expected
    assert len(tables["landsat_nemenyi"]) == 4 * 45 and len(tables["pcap_vs_random"]) == 2 * 3 * 10
    assert len(tables["pcap_hit_rates"]) == 2 * 3 * 10
    assert messages[-1].startswith("Saved 6 tables")


def test_run_statistical_tests_writes_nothing_when_save_is_false(tmp_path):
    run_statistical_tests(save=False, out_dir=tmp_path / "none", show=lambda message: None)
    assert not (tmp_path / "none").exists()


def test_cd_groups_join_only_ranks_within_the_critical_difference():
    assert cd_groups([1.0, 1.5, 2.0, 5.0, 5.2], 1.0) == [(0, 2), (3, 4)]
    assert cd_groups([1.0, 3.0, 5.0], 1.0) == []


def test_cd_groups_drop_groups_inside_an_earlier_group():
    assert cd_groups([1.0, 2.0, 2.5, 3.5], 2.0) == [(0, 2), (1, 3)]


def test_save_cd_diagrams_writes_one_png_per_task(tmp_path):
    run_statistical_tests(out_dir=tmp_path, show=lambda message: None)
    paths = save_cd_diagrams(tmp_path / "landsat_ranks.csv", tmp_path, show=lambda message: None)
    assert [p.name for p in paths] == ["cd_landsat_fs.png", "cd_landsat_threshold.png",
                                       "cd_landsat_components.png", "cd_landsat_joint.png"]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_fisher_matches_scipy_and_applies_holm():
    from scipy import stats
    table = fisher_against_baseline({"RANDOM": 8, "GWO": 16, "SMA": 2}, {"RANDOM": 20, "GWO": 20, "SMA": 20})
    assert list(table.algorithm) == ["GWO", "SMA"]
    assert table.loc[0, "hit_rate"] == 0.8 and table.loc[0, "difference"] == pytest.approx(0.4)
    assert table.loc[0, "p_value"] == pytest.approx(stats.fisher_exact([[16, 4], [8, 12]])[1])
    assert (table.p_holm >= table.p_value).all()


def test_fisher_refuses_a_missing_baseline():
    with pytest.raises(ValueError, match="Baseline"):
        fisher_against_baseline({"GWO": 3}, {"GWO": 20})


def test_pcap_hit_rates_compare_ten_algorithms_with_twenty_runs_each():
    table = pcap_hit_rates("grid", 400)
    assert len(table) == 10 and (table.runs == 20).all() and (table.baseline_runs == 20).all()
    gwo = table[table.algorithm == "GWO"].iloc[0]
    assert (gwo.hits, gwo.baseline_hits) == (16, 8)
