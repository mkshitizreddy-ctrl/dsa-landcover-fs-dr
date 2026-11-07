import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from meta_fs import ALGORITHMS
from pcap_feature_selection import FS_DIR, METHODS, THRESHOLDS
from pcap_grid import CLASSIFIERS, SCALERS
from pcap_mh_search import (RANDOM, GridSpace, append_rows, check_algorithms, p_random_hit, pick_option,
                            run_algorithm, run_random, summarize_runs)

ROOT = Path(__file__).resolve().parents[1]
FS_SUMMARY_PATH = FS_DIR / "fs_summary.csv"
FS_SEARCH_DIR = ROOT / "results" / "pcap_fs_search_v1"


class SelectionSpace(GridSpace):

    def __init__(self, table, n_features=36, metric="f1_macro_mean"):
        self.n_features = n_features
        self.score = {(r.scaler, r.method, float(r.param), r.classifier): float(getattr(r, metric))
                      for r in table.itertuples()}
        expected = len(SCALERS) * len(CLASSIFIERS) * (n_features + len(THRESHOLDS))
        if len(self.score) != expected:
            raise ValueError(f"Selection table has {len(self.score)} settings, expected {expected}; "
                             "run the full feature-selection study first")
        self.sorted_scores = np.sort(np.array(list(self.score.values())))
        self.n = len(self.sorted_scores)
        self.best = float(self.sorted_scores[-1])
        self.best_cfg = max(self.score, key=self.score.get)

    def decode(self, position):
        p = np.asarray(position, dtype=float)
        method = pick_option(p[1], METHODS)
        if method == "MI_top":
            param = float(min(self.n_features, max(1, math.ceil(p[2] * self.n_features - 1e-9))))
        else:
            param = pick_option(p[2], THRESHOLDS)
        return (pick_option(p[0], SCALERS), method, param, pick_option(p[3], CLASSIFIERS))


def search_odds(space, near=0.01, draws=(25, 50, 100, 200, 400)):
    n_best, n_near = space.count_within(0.0), space.count_within(near)
    return pd.DataFrame({
        "configs_tried": list(draws),
        "finds_best_%": [100 * p_random_hit(space.n, n_best, n) for n in draws],
        "finds_near_best_%": [100 * p_random_hit(space.n, n_near, n) for n in draws],
    })


def selection_labels(row):
    names = {"reduction": "method", "k": "param"}
    return {names.get(key, key): value for key, value in row.items()}


def run_selection_search(summary_path=FS_SUMMARY_PATH, algorithms=tuple(ALGORITHMS) + (RANDOM,),
                         budgets=(100, 200, 400), n_seeds=20, population=20, near=0.01, n_features=36, save=True,
                         out_dir=FS_SEARCH_DIR, show=print):
    check_algorithms(algorithms)
    space = SelectionSpace(pd.read_csv(summary_path), n_features=n_features)
    show(f"Settings: {space.n} | best {space.best:.4f} = {space.best_cfg} | "
         f"tied with the best: {space.count_within(0.0)} | within {near} of the best: {space.count_within(near)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started, runs = time.perf_counter(), []
    for budget in budgets:
        for name in algorithms:
            for seed in range(1, n_seeds + 1):
                if name == RANDOM:
                    result = run_random(space, budget, seed, near)
                else:
                    result = run_algorithm(name, space, budget, population, seed, near)
                runs.append(selection_labels({"run_id": run_id, "population": population, "near": near, **result}))
            show(f"budget {budget} {name:6} done | elapsed {time.perf_counter() - started:.0f}s")

    summary = summarize_runs(runs)
    odds = search_odds(space, near)
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        append_rows(out_dir / "runs.csv", runs)
        summary.to_csv(out_dir / f"summary_{run_id}.csv", index=False)
        odds.to_csv(out_dir / "random_search_odds.csv", index=False)
        show(f"Saved {out_dir / 'runs.csv'}")
    return pd.DataFrame(runs), summary, odds
