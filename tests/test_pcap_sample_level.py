import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcap_sample_level import (SampleTables, aggregate_by_hash, choose_classifier, compare_with_packets,
                               cross_validate_samples, hash_folds, run_sample_cv, run_sample_final_test,
                               sample_counts, score_predictions, summarize_cv)


def packets():
    X = pd.DataFrame({"a": [1.0, 3.0, 5.0, 10.0, np.nan, 7.0], "b": [0.0, 0.0, 6.0, 2.0, 4.0, 1.0]})
    y = np.array([2, 2, 2, 0, 0, 1])
    groups = np.array(["h1", "h1", "h1", "h2", "h2", "h3"])
    return X, y, groups


def test_aggregate_gives_one_row_per_hash_with_every_statistic():
    table, labels = aggregate_by_hash(*packets())
    assert list(table.index) == ["h1", "h2", "h3"]
    assert list(table.columns) == ["a_mean", "a_std", "a_min", "a_max", "b_mean", "b_std", "b_min", "b_max",
                                   "packets"]
    assert table.loc["h1", ["a_mean", "a_min", "a_max", "packets"]].tolist() == [3.0, 1.0, 5.0, 3]
    assert table.loc["h1", "a_std"] == pytest.approx(pd.Series([1.0, 3.0, 5.0]).std())
    assert labels.tolist() == [2, 0, 1]


def test_aggregate_ignores_missing_values_and_gives_one_packet_zero_spread():
    table, _ = aggregate_by_hash(*packets())
    assert table.loc["h2", ["a_mean", "a_min", "a_max"]].tolist() == [10.0, 10.0, 10.0]
    assert table.loc["h3", ["a_std", "b_std"]].tolist() == [0.0, 0.0]


def test_aggregate_matches_a_direct_groupby():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x", "y", "z"])
    groups = rng.integers(0, 15, 200).astype(str)
    y = np.array([int(g) % 3 for g in groups])
    table, labels = aggregate_by_hash(X, y, groups)
    expected = X.assign(Hash=groups).groupby("Hash").max()
    assert np.allclose(table[["x_max", "y_max", "z_max"]].to_numpy(), expected.to_numpy())
    assert labels.tolist() == [int(g) % 3 for g in table.index]
    assert table.packets.sum() == 200


def test_aggregate_refuses_a_hash_with_two_families():
    X, y, groups = packets()
    y[1] = 0
    with pytest.raises(ValueError, match="more than one family"):
        aggregate_by_hash(X, y, groups)


def test_hash_folds_maps_each_hash_to_its_validation_fold():
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])), (np.array([0, 1, 5]), np.array([2, 3, 4]))]
    assert hash_folds(groups, folds) == {"a": 0, "b": 1, "c": 1}


def test_hash_folds_refuses_a_hash_in_two_folds():
    groups = np.array(["a", "a", "b"])
    folds = [(np.array([1, 2]), np.array([0])), (np.array([0, 2]), np.array([1]))]
    with pytest.raises(ValueError, match="more than one validation fold"):
        hash_folds(groups, folds)


def make_tables(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(3), n // 3)
    X = pd.DataFrame(rng.normal(size=(n, 4)) + y[:, None] * 6.0, columns=["a", "b", "c", "d"])
    X.iloc[0, 0] = np.nan
    folds = np.arange(n) % 4
    return SampleTables(X_dev=X, y_dev=y, fold_dev=folds, X_test=X.iloc[:30], y_test=y[:30],
                        class_names=["A", "B", "C"])


def test_score_predictions_gives_overall_and_per_family_scores():
    scores = score_predictions(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), ["A", "B", "C"])
    assert scores["accuracy"] == 0.5 and scores["f1_A"] == pytest.approx(2 / 3)
    assert scores["f1_C"] == 0.0 and scores["f1_macro"] == pytest.approx((2 / 3 + 0.5 + 0) / 3)


def test_cross_validation_scores_every_classifier_on_every_fold():
    scores = cross_validate_samples(make_tables(), classifiers=("DT", "RF"), show=lambda message: None)
    assert len(scores) == 8 and set(scores.fold) == {1, 2, 3, 4}
    assert (scores.train_samples + scores.valid_samples == 240).all()
    assert (scores.f1_macro > 0.9).all()


def test_summary_averages_the_folds_and_puts_the_best_first():
    folds = pd.DataFrame({"classifier": ["X", "X", "Y", "Y"], "fold": [1, 2, 1, 2], "fit_seconds": 0.1,
                          "accuracy": [0.5, 0.7, 0.9, 0.9], "balanced_accuracy": 0.5,
                          "f1_macro": [0.2, 0.4, 0.6, 0.8], "f1_A": [0.1, 0.3, 0.5, 0.5]})
    summary = summarize_cv(folds)
    assert list(summary.classifier) == ["Y", "X"]
    assert summary.loc[0, "f1_macro"] == pytest.approx(0.7) and summary.loc[1, "accuracy"] == pytest.approx(0.6)
    assert summary.loc[0, "f1_macro_sd"] == pytest.approx(np.std([0.6, 0.8], ddof=1))


def test_run_sample_cv_saves_three_tables_and_never_scores_the_test_samples(tmp_path):
    tables = make_tables()
    tables.X_test = tables.X_test * np.nan
    summary = run_sample_cv(classifiers=("DT",), out_dir=tmp_path, tables=tables, show=lambda message: None)
    assert {p.name for p in tmp_path.iterdir()} == {"cv_folds.csv", "cv_summary.csv", "samples.csv"}
    assert summary.classifier.tolist() == ["DT"]
    assert sample_counts(tables).test_samples.sum() == 30


def test_choose_classifier_takes_the_best_cv_macro_f1():
    summary = pd.DataFrame({"classifier": ["DT", "RF", "SVM"], "f1_macro": [0.28, 0.29, 0.27]})
    assert choose_classifier(summary) == ("RF", 0.29)


def test_compare_with_packets_uses_the_configuration_chosen_by_cv_macro_f1():
    sample = {"classifier": "RF", "cv_f1_macro": 0.29, "f1_macro": 0.31, "balanced_accuracy": 0.35,
              "accuracy": 0.74}
    table = compare_with_packets(sample)
    assert table.level.tolist() == ["packet", "sample"]
    assert table.configuration.iloc[0] == "Normalizer + PCA (14) + DT"
    assert table.test_f1_macro.tolist() == pytest.approx([0.298698, 0.31], abs=1e-6)


def test_final_test_runs_once_and_saves_every_output(tmp_path):
    tables = make_tables()
    cv_path = tmp_path / "cv_summary.csv"
    pd.DataFrame({"classifier": ["SVM", "DT"], "f1_macro": [0.5, 0.9]}).to_csv(cv_path, index=False)
    out = tmp_path / "final"
    comparison = run_sample_final_test(cv_summary_path=cv_path, out_dir=out, tables=tables, show=lambda m: None)
    assert {p.name for p in out.iterdir()} == {"test_results.csv", "test_report_sample_DT.txt",
                                               "test_confusion_sample_DT.csv", "comparison_with_packets.csv"}
    result = pd.read_csv(out / "test_results.csv").iloc[0]
    assert result.classifier == "DT" and result.test_samples == 30 and result.f1_macro > 0.9
    assert comparison.level.tolist() == ["packet", "sample"]
    with pytest.raises(FileExistsError, match="already been used"):
        run_sample_final_test(cv_summary_path=cv_path, out_dir=out, tables=tables, show=lambda m: None)
