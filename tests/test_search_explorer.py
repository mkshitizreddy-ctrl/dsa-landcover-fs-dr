import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from pcap_mh_search import RANDOM
from search_explorer import compare_algorithms, describe, load_space, search_with_curve


@pytest.mark.parametrize("kind, size", [("grid", 924), ("selection", 1644)])
def test_both_saved_spaces_load(kind, size):
    assert load_space(kind).n == size


def test_unknown_space_is_refused():
    with pytest.raises(ValueError, match="Unknown space"):
        load_space("landsat")


def test_descriptions_name_the_setting():
    assert describe(("Normalizer", "PCA", 14, "DT"), "grid") == "Normalizer + PCA, k = 14 + DT"
    assert describe(("Normalizer", "HeapBFS", 0.58, "DT"), "selection") == "Normalizer + heap + BFS, t = 0.58 + DT"
    assert describe(("Standard", "MI_top", 22.0, "XGB"), "selection") == "Standard + MI top-k, k = 22 + XGB"


@pytest.mark.parametrize("name", [RANDOM, "PSO"])
def test_curve_rises_and_matches_the_result(name):
    space = load_space("grid")
    result, curve = search_with_curve(name, space, budget=60, population=10, seed=3)
    assert len(curve) == result["unique_configs"] and result["calls"] >= result["unique_configs"]
    assert np.all(np.diff(curve.best_score) >= 0) and curve.best_score.iloc[-1] == result["best_f1"]


def test_compare_runs_every_requested_algorithm():
    summary = compare_algorithms("selection", budget=30, n_seeds=2, show=lambda message: None)
    assert len(summary) == 11 and "budget" not in summary.columns
