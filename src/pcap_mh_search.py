import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mealpy import FloatVar

from meta_fs import ALGORITHM_FAMILY, ALGORITHMS, build_optimizer
from pcap_grid import CLASSIFIERS, GRID_DIR, REDUCTIONS, SCALERS, k_values

ROOT = Path(__file__).resolve().parents[1]
GRID_SUMMARY_PATH = GRID_DIR / "grid_summary.csv"
SEARCH_DIR = ROOT / "results" / "pcap_mh_v1"
RANDOM = "RANDOM"
CHECKPOINTS = (10, 25, 50, 100, 200, 400)


def pick_option(x, options):
    return options[min(int(float(x) * len(options)), len(options) - 1)]


class GridSpace:

    def __init__(self, table, n_features=36, n_classes=6, metric="f1_macro_mean"):
        self.score = {(r.scaler, r.reduction, int(r.k), r.classifier): float(getattr(r, metric))
                      for r in table.itertuples()}
        self.k_top = {red: max(k_values(red, n_features, n_classes)) for red in REDUCTIONS}
        expected = (sum(len(k_values(r, n_features, n_classes)) for r in REDUCTIONS)
                    * len(SCALERS) * len(CLASSIFIERS))
        if len(self.score) != expected:
            raise ValueError(f"Grid table has {len(self.score)} configurations, expected {expected}; "
                             "run the full grid first")
        self.sorted_scores = np.sort(np.array(list(self.score.values())))
        self.n = len(self.sorted_scores)
        self.best = float(self.sorted_scores[-1])
        self.best_cfg = max(self.score, key=self.score.get)

    def rank(self, cfg):
        return int(self.n - np.searchsorted(self.sorted_scores, self.score[cfg], side="right") + 1)

    def count_within(self, gap):
        return int((self.sorted_scores >= self.best - gap - 1e-12).sum())

    def decode(self, position):
        p = np.asarray(position, dtype=float)
        reduction = pick_option(p[1], REDUCTIONS)
        top = self.k_top[reduction]
        k = int(min(top, max(1, math.ceil(p[2] * top - 1e-9))))
        return (pick_option(p[0], SCALERS), reduction, k, pick_option(p[3], CLASSIFIERS))


class Tracker:

    def __init__(self, space, near=0.01):
        self.space, self.near = space, near
        self.seen = {}
        self.calls = 0
        self.best_cfg, self.best_score = None, -math.inf
        self.unique_to_best = self.unique_to_near = None
        self.curve = []

    def __call__(self, position):
        self.calls += 1
        cfg = self.space.decode(position)
        if cfg not in self.seen:
            self.record_new(cfg)
        return 1.0 - self.seen[cfg]

    def record_new(self, cfg):
        score = self.space.score[cfg]
        self.seen[cfg] = score
        if score > self.best_score:
            self.best_score, self.best_cfg = score, cfg
        self.curve.append(self.best_score)
        tried = len(self.seen)
        if self.unique_to_near is None and self.best_score >= self.space.best - self.near - 1e-12:
            self.unique_to_near = tried
        if self.unique_to_best is None and self.best_score >= self.space.best - 1e-12:
            self.unique_to_best = tried

    def result(self, algorithm, budget, seed, seconds):
        cfg = self.best_cfg
        row = {
            "algorithm": algorithm, "family": ALGORITHM_FAMILY.get(algorithm, "baseline"),
            "budget": budget, "seed": seed, "calls": self.calls, "unique_configs": len(self.seen),
            "best_f1": self.best_score, "best_rank": self.space.rank(cfg),
            "gap_to_best": self.space.best - self.best_score,
            "hit_best": self.unique_to_best is not None, "hit_near": self.unique_to_near is not None,
            "unique_to_best": self.unique_to_best, "unique_to_near": self.unique_to_near,
            "scaler": cfg[0], "reduction": cfg[1], "k": cfg[2], "classifier": cfg[3],
            "seconds": round(seconds, 3),
        }
        for c in CHECKPOINTS:
            row[f"best_at_{c}"] = self.curve[min(c, len(self.curve)) - 1]
        return row


def run_algorithm(name, space, budget, population, seed, near=0.01):
    tracker = Tracker(space, near)
    problem = {"obj_func": tracker, "bounds": FloatVar(lb=[0.0] * 4, ub=[1.0] * 4),
               "minmax": "min", "log_to": None}
    optimizer = build_optimizer(name, 10_000, population)
    start = time.perf_counter()
    optimizer.solve(problem, seed=seed, termination={"max_fe": int(budget)})
    return tracker.result(name, budget, seed, time.perf_counter() - start)


def run_random(space, budget, seed, near=0.01):
    tracker = Tracker(space, near)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for _ in range(budget):
        tracker(rng.random(4))
    return tracker.result(RANDOM, budget, seed, time.perf_counter() - start)


def p_random_hit(n_total, n_good, n_draws):
    p_miss = 1.0
    for i in range(min(n_draws, n_total)):
        p_miss *= max(0.0, (n_total - n_good - i) / (n_total - i))
    return 1.0 - p_miss


def random_search_odds(space, near=0.01, draws=(25, 50, 100, 200)):
    n_near = space.count_within(near)
    return pd.DataFrame({
        "configs_tried": list(draws),
        "finds_best_%": [100 * p_random_hit(space.n, 1, n) for n in draws],
        "finds_near_best_%": [100 * p_random_hit(space.n, n_near, n) for n in draws],
    })


def append_rows(path, rows):
    fields = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != fields:
                raise ValueError(f"Refusing to append a different schema to {path}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize_runs(runs):
    groups = pd.DataFrame(runs).groupby(["budget", "algorithm"])
    summary = pd.DataFrame({
        "hit_best_%": groups.hit_best.mean() * 100,
        "hit_near_%": groups.hit_near.mean() * 100,
        "mean_best_f1": groups.best_f1.mean(),
        "mean_rank": groups.best_rank.mean(),
        "median_rank": groups.best_rank.median(),
        "unique_configs": groups.unique_configs.mean(),
        "median_unique_to_near": groups.unique_to_near.median(),
    }).reset_index()
    return summary.sort_values(["budget", "hit_near_%", "mean_rank"], ascending=[True, False, True])


def check_algorithms(algorithms):
    unknown = [a for a in algorithms if a not in ALGORITHMS and a != RANDOM]
    if unknown:
        raise ValueError(f"Unknown algorithms {unknown}")


def run_search(grid_path=GRID_SUMMARY_PATH, algorithms=tuple(ALGORITHMS) + (RANDOM,), budgets=(100, 200, 400),
               n_seeds=20, population=20, near=0.01, save=True, out_dir=SEARCH_DIR, show=print):
    check_algorithms(algorithms)
    space = GridSpace(pd.read_csv(grid_path))
    show(f"Grid: {space.n} configurations | best {space.best:.4f} = {space.best_cfg} | "
         f"within {near} of the best: {space.count_within(near)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started, runs = time.perf_counter(), []
    for budget in budgets:
        for name in algorithms:
            for seed in range(1, n_seeds + 1):
                if name == RANDOM:
                    result = run_random(space, budget, seed, near)
                else:
                    result = run_algorithm(name, space, budget, population, seed, near)
                runs.append({"run_id": run_id, "population": population, "near": near, **result})
            show(f"budget {budget} {name:6} done | elapsed {time.perf_counter() - started:.0f}s")

    summary = summarize_runs(runs)
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        append_rows(out_dir / "runs.csv", runs)
        summary.to_csv(out_dir / f"summary_{run_id}.csv", index=False)
        show(f"Saved {out_dir / 'runs.csv'}")
    return pd.DataFrame(runs), summary
