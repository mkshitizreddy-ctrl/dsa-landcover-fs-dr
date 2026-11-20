import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from results_explorer import (BEST_CONFUSION_PATH, best_by_k, cd_figures, class_counts, confusion_files,
                              describe_setting, filter_grid, load_grid, majority_accuracy, read_confusion,
                              final_test_comparison, row_percentages, split_summary, stats_tables, top_configs,
                              weighting_table)


def test_split_summary_reads_the_saved_splits():
    summary = split_summary()
    sizes = dict(zip(summary["table"].part, summary["table"].packets))
    assert sizes["Test, used once"] == 145355 and sizes["Development"] == 591484
    assert summary["families"] == ["Benign", "DarkNexus", "Gafgyt", "Generic", "Mirai", "Unknown"]
    assert len(summary["features"]) == 36 and summary["constant_columns"] == ["SMTP", "SSH", "IRC"]


def test_best_confusion_matrix_counts_every_test_packet():
    confusion = read_confusion(BEST_CONFUSION_PATH)
    assert class_counts(confusion).sum() == 145355
    assert majority_accuracy(confusion) == pytest.approx(0.699, abs=0.001)


def test_row_percentages_sum_to_one_hundred_and_survive_an_empty_row():
    confusion = pd.DataFrame([[3, 1], [0, 0]], index=["a", "b"], columns=["a", "b"])
    percent = row_percentages(confusion)
    assert percent.loc["a"].tolist() == [75.0, 25.0] and percent.loc["b"].tolist() == [0.0, 0.0]


def test_every_saved_confusion_matrix_is_listed():
    files = confusion_files()
    assert len(files) == 16 and "Normalizer PCA k14 DT" in files


def test_grid_filters_and_ranks_the_924_configurations():
    grid = load_grid()
    assert len(grid) == 924
    dt = filter_grid(grid, classifiers=["DT"])
    assert set(dt.classifier) == {"DT"} and len(filter_grid(grid)) == 924
    top = top_configs(grid, n=3)
    assert len(top) == 3 and top.f1_macro_mean.is_monotonic_decreasing
    assert tuple(top.iloc[0][["scaler", "reduction", "k", "classifier"]]) == ("Normalizer", "PCA", 14, "DT")


def test_best_by_k_keeps_one_score_per_reduction_and_k():
    curve = best_by_k(load_grid())
    assert not curve.duplicated(["reduction", "k"]).any()
    assert set(curve.reduction) == {"PCA", "KernelPCA", "LDA"}


@pytest.mark.parametrize("method, param, text", [("MI_top", 33.0, "MI top-k, k = 33"),
                                                 ("HeapBFS", 0.58, "heap + BFS, t = 0.58"),
                                                 ("Mask", "GA seed 2", "metaheuristic mask, GA seed 2"),
                                                 ("PCA", "14", "PCA, k = 14")])
def test_settings_are_described_in_words(method, param, text):
    assert describe_setting(method, param) == text


def test_final_comparison_puts_selection_and_extraction_side_by_side():
    table = final_test_comparison()
    assert len(table) == 16 and set(table.family) == {"selection", "extraction"}
    assert table.test_f1_macro.is_monotonic_decreasing


def test_weighting_table_has_one_column_per_scheme():
    table = weighting_table()
    assert list(table.columns[-3:]) == ["none", "sqrt", "balanced"] and table[["none", "sqrt", "balanced"]].notna().all().all()


def test_statistical_results_are_found():
    assert {"landsat_friedman", "pcap_vs_random"} <= set(stats_tables())
    assert len(cd_figures()) == 4
