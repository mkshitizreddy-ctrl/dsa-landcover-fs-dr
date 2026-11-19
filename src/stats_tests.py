from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LANDSAT_MH_PATH = ROOT / "results" / "mh_comparison_v1.csv"
ALGORITHMS = ("GA", "DE", "PSO", "GWO", "WOA", "HHO", "ABC", "MFO", "SSA", "SMA")
LANDSAT_TASKS = ("fs", "threshold", "components", "joint")


def score_matrix(runs, block_columns, algorithm_column, score_column, algorithms=None):
    if algorithms is not None:
        runs = runs[runs[algorithm_column].isin(algorithms)]
    table = runs.pivot_table(index=block_columns, columns=algorithm_column, values=score_column, aggfunc="mean")
    if table.empty or table.isna().any().any():
        raise ValueError("Every algorithm needs a score in every block")
    return table


def average_ranks(table, higher_is_better=True):
    scores = -table if higher_is_better else table
    return scores.rank(axis=1, method="average").mean().sort_values()


def friedman_test(table):
    if table.shape[1] < 3:
        raise ValueError("The Friedman test needs at least three algorithms")
    if table.shape[0] < 2:
        raise ValueError("The Friedman test needs at least two blocks")
    statistic, p_value = stats.friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    return {"n_blocks": int(table.shape[0]), "n_algorithms": int(table.shape[1]),
            "statistic": float(statistic), "p_value": float(p_value)}


def landsat_friedman(path=LANDSAT_MH_PATH, score_column="macro_f1", tasks=LANDSAT_TASKS):
    runs = pd.read_csv(path)
    rows = []
    for task in tasks:
        table = score_matrix(runs[runs.task == task], ["seed", "outer_fold"], "method", score_column, ALGORITHMS)
        ranks = average_ranks(table)
        rows.append({"task": task, **friedman_test(table),
                     "best_algorithm": ranks.index[0], "best_mean_rank": float(ranks.iloc[0])})
    return pd.DataFrame(rows)
