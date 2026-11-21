import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from make_report_figures import growth, make_report_figures


def test_growth_recovers_a_known_exponent():
    table = pd.DataFrame({"n": [100, 200, 400, 800], "t": [1.0, 4.0, 16.0, 64.0]})
    assert abs(growth(table, "n", "t", 100) - 2.0) < 1e-9


def test_every_figure_and_table_is_written(tmp_path):
    paths = make_report_figures(tmp_path, show=lambda message: None)
    names = sorted(p.name for p in paths)
    assert len(paths) == 13 and all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert sum(name.endswith(".png") for name in names) == 9 and sum(name.endswith(".csv") for name in names) == 4
    complexity = pd.read_csv(tmp_path / "T1_complexity.csv")
    assert list(complexity.Stage)[0] == "Mutual information" and len(complexity) == 5
    datasets = pd.read_csv(tmp_path / "T2_datasets.csv")
    assert list(datasets["Rows used"]) == [6435, 736839]
    assert list(datasets["Test rows"]) == [2000, 145355] and list(datasets["Features used"]) == [36, 36]
