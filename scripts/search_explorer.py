import time

import numpy as np
import pandas as pd
from mealpy import FloatVar

from meta_fs import ALGORITHMS, build_optimizer
from pcap_fs_search import FS_SUMMARY_PATH, SelectionSpace, run_selection_search
from pcap_mh_search import GRID_SUMMARY_PATH, RANDOM, GridSpace, Tracker, run_search

SPACES = {"pcap grid: scaler, reduction, k, classifier (924 configurations)": "grid",
          "pcap selection: scaler, MI top-k or heap + BFS t, classifier (1,644 settings)": "selection"}
CHOICES = tuple(ALGORITHMS) + (RANDOM,)


def load_space(kind):
    if kind == "grid":
        return GridSpace(pd.read_csv(GRID_SUMMARY_PATH))
    if kind == "selection":
        return SelectionSpace(pd.read_csv(FS_SUMMARY_PATH))
    raise ValueError(f"Unknown space {kind!r}; choose from {list(SPACES.values())}")


def describe(cfg, kind):
    scaler, method, param, clf = cfg
    if kind == "selection" and method == "HeapBFS":
        setting = f"heap + BFS, t = {param:.2f}"
    elif kind == "selection":
        setting = f"MI top-k, k = {int(param)}"
    else:
        setting = f"{method}, k = {int(param)}"
    return f"{scaler} + {setting} + {clf}"


def search_with_curve(name, space, budget, population=20, seed=1, near=0.01):
    tracker = Tracker(space, near)
    start = time.perf_counter()
    if name == RANDOM:
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            tracker(rng.random(4))
    else:
        problem = {"obj_func": tracker, "bounds": FloatVar(lb=[0.0] * 4, ub=[1.0] * 4), "minmax": "min",
                   "log_to": None}
        build_optimizer(name, 10_000, population).solve(problem, seed=seed, termination={"max_fe": int(budget)})
    result = tracker.result(name, budget, seed, time.perf_counter() - start)
    curve = pd.DataFrame({"configs_tried": np.arange(1, len(tracker.curve) + 1), "best_score": tracker.curve})
    return result, curve


def compare_algorithms(kind, budget, n_seeds=10, population=20, near=0.01, show=print):
    if kind == "grid":
        _, summary = run_search(algorithms=CHOICES, budgets=(budget,), n_seeds=n_seeds, population=population,
                                near=near, save=False, show=show)
    elif kind == "selection":
        _, summary, _ = run_selection_search(algorithms=CHOICES, budgets=(budget,), n_seeds=n_seeds,
                                             population=population, near=near, save=False, show=show)
    else:
        raise ValueError(f"Unknown space {kind!r}; choose from {list(SPACES.values())}")
    return summary.drop(columns="budget").reset_index(drop=True)
