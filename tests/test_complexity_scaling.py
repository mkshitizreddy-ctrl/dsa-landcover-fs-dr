import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_complexity import STAGE_COLUMNS, growth_exponents, run_data_scaling, time_stages


def test_time_stages_reports_every_stage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    X = np.hstack([y[:, None] + rng.normal(size=(300, 1)), rng.normal(size=(300, 5))])
    X[:, 5] = 1.0
    row = time_stages(X, y, threshold=0.5, trials=1)
    assert row["n"] == 300 and row["d"] == 6 and 1 <= row["kept"] <= 6
    assert all(row[c] >= 0 for c in STAGE_COLUMNS)


def test_growth_exponent_of_a_linear_and_a_quadratic_stage():
    n = np.array([1000, 2000, 4000, 8000])
    results = pd.DataFrame({"n": n, "mi_s": n * 1e-6, "corr_s": (n / 1000.0) ** 2})
    exponents = growth_exponents(results, min_d=1000, size_column="n", columns=("mi_s", "corr_s"))
    assert exponents["mi_s"] == pytest.approx(1.0) and exponents["corr_s"] == pytest.approx(2.0)


def test_landsat_scaling_runs_on_small_row_counts():
    results, exponents, _ = run_data_scaling("landsat", rows=(100, 200), trials=1, show=lambda message: None)
    assert list(results.n) == [100, 200] and set(results.d) == {36}
    assert set(exponents) <= set(STAGE_COLUMNS)


def test_unknown_dataset_is_refused():
    with pytest.raises(ValueError, match="Unknown dataset"):
        run_data_scaling("mnist")
