"""Stage 3: metaheuristic search over the pcap pipeline grid, scored by lookup.

The exhaustive grid (pcap_grid.py) already scored all 924 configurations
(scaler x reduction x k x classifier) with 4-fold CV. Here each Mealpy
algorithm searches the same space, and a candidate's fitness is LOOKED UP in
grid_summary.csv instead of retraining models. The grid therefore gives the
true optimum, and we can measure how quickly each algorithm gets there.

Encoding: a position in [0, 1]^4 decodes to
  scaler      = floor(x0 * 3)        (Standard, MinMax, Normalizer)
  reduction   = floor(x1 * 3)        (PCA, KernelPCA, LDA)
  k           = ceil(x2 * k_max)     (k_max = 36, or 5 for LDA)
  classifier  = floor(x3 * 4)        (RF, XGB, DT, SVM)
Fitness (minimised) = 1 - CV macro-F1 of that configuration.

Cost: in a real search every NEW configuration costs 4 model fits, and a
repeated one costs nothing (cached), so cost = number of distinct
configurations evaluated. Each run gets the same budget of fitness calls.

Baseline: RANDOM samples uniform positions with the same decoding and budget.
A metaheuristic is only useful if it beats random search.

Run from the repository root:
    python src\\pcap_mh_search.py
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mealpy import FloatVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_fs import ALGORITHM_FAMILY, ALGORITHMS, build_optimizer  # noqa: E402
from pcap_grid import CLASSIFIERS, REDUCTIONS, SCALERS, k_values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (10, 25, 50, 100, 200, 400)


def _pick(x, options):
    """x in [0, 1] -> one of the options, equal-width cells."""
    return options[min(int(float(x) * len(options)), len(options) - 1)]


class GridSpace:
    """Lookup table over every configuration, with ranks and decoding."""

    def __init__(self, table, n_features=36, n_classes=6, metric="f1_macro_mean"):
        self.score = {(r.scaler, r.reduction, int(r.k), r.classifier): float(getattr(r, metric))
                      for r in table.itertuples()}
        self.k_top = {red: max(k_values(red, n_features, n_classes)) for red in REDUCTIONS}
        expected = (sum(len(k_values(r, n_features, n_classes)) for r in REDUCTIONS)
                    * len(SCALERS) * len(CLASSIFIERS))
        if len(self.score) != expected:
            raise ValueError(f"Grid table has {len(self.score)} configurations, expected {expected}; "
                             "run the full grid first")
        values = np.sort(np.array(list(self.score.values())))           # ascending
        self.n = len(values)
        self.best = float(values[-1])
        self.best_cfg = max(self.score, key=self.score.get)
        self._sorted = values

    def rank(self, cfg):
        """1 = best; ties share the better rank."""
        return int(self.n - np.searchsorted(self._sorted, self.score[cfg], side="right") + 1)

    def count_within(self, gap):
        return int((self._sorted >= self.best - gap - 1e-12).sum())

    def decode(self, position):
        p = np.asarray(position, dtype=float)
        red = _pick(p[1], REDUCTIONS)
        top = self.k_top[red]
        k = int(min(top, max(1, math.ceil(p[2] * top - 1e-9))))
        return (_pick(p[0], SCALERS), red, k, _pick(p[3], CLASSIFIERS))


class Tracker:
    """Fitness function that also records the search's progress."""

    def __init__(self, space, near=0.01):
        self.space, self.near = space, near
        self.seen = {}                    # configuration -> score (the cache)
        self.calls = 0
        self.best_cfg, self.best_score = None, -math.inf
        self.unique_to_best = self.unique_to_near = None
        self.curve = []                   # best score after each NEW configuration

    def __call__(self, position):
        self.calls += 1
        cfg = self.space.decode(position)
        if cfg not in self.seen:
            s = self.space.score[cfg]
            self.seen[cfg] = s
            if s > self.best_score:
                self.best_score, self.best_cfg = s, cfg
            self.curve.append(self.best_score)
            n = len(self.seen)
            if self.unique_to_near is None and self.best_score >= self.space.best - self.near - 1e-12:
                self.unique_to_near = n
            if self.unique_to_best is None and self.best_score >= self.space.best - 1e-12:
                self.unique_to_best = n
        return 1.0 - self.seen[cfg]

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
    t0 = time.perf_counter()
    optimizer.solve(problem, seed=seed, termination={"max_fe": int(budget)})
    return tracker.result(name, budget, seed, time.perf_counter() - t0)


def run_random(space, budget, seed, near=0.01):
    tracker = Tracker(space, near)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(budget):
        tracker(rng.random(4))
    return tracker.result("RANDOM", budget, seed, time.perf_counter() - t0)


def p_random_hit(n_total, n_good, n_draws):
    """Chance that n_draws distinct configurations include at least one good one."""
    p_miss = 1.0
    for i in range(min(n_draws, n_total)):
        p_miss *= max(0.0, (n_total - n_good - i) / (n_total - i))
    return 1.0 - p_miss


def append_rows(path, rows):
    fields = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != fields:
                raise ValueError(f"Refusing to append a different schema to {path}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", default=str(ROOT / "results" / "pcap_grid_v1" / "grid_summary.csv"))
    p.add_argument("--algorithms", default=",".join(list(ALGORITHMS) + ["RANDOM"]))
    p.add_argument("--budgets", default="100,200,400", help="fitness calls per run")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--near", type=float, default=0.01, help="'near-best' = within this macro-F1 of the best")
    p.add_argument("--out", default=str(ROOT / "results" / "pcap_mh_v1"))
    return p.parse_args()


def main():
    args = parse_args()
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    bad = [a for a in algos if a not in ALGORITHMS and a != "RANDOM"]
    if bad:
        raise ValueError(f"Unknown algorithms {bad}")
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = list(range(1, args.n_seeds + 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    space = GridSpace(pd.read_csv(args.grid))
    n_near = space.count_within(args.near)
    print(f"Grid: {space.n} configurations | best {space.best:.4f} = {space.best_cfg}")
    print(f"Configurations within {args.near} of the best: {n_near}")
    for n in (25, 50, 100, 200):
        print(f"  random pick of {n:3} distinct configs finds the best: {p_random_hit(space.n, 1, n):.1%}, "
              f"a near-best one: {p_random_hit(space.n, n_near, n):.1%}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started, rows = time.perf_counter(), []
    for budget in budgets:
        for name in algos:
            for seed in seeds:
                res = (run_random(space, budget, seed, args.near) if name == "RANDOM"
                       else run_algorithm(name, space, budget, args.population, seed, args.near))
                rows.append({"run_id": run_id, "population": args.population, "near": args.near, **res})
            print(f"budget {budget} {name:6} done | elapsed {time.perf_counter() - started:.0f}s", flush=True)
    append_rows(out / "runs.csv", rows)

    df = pd.DataFrame(rows)
    g = df.groupby(["budget", "algorithm"])
    summary = pd.DataFrame({
        "hit_best_%": g.hit_best.mean() * 100,
        "hit_near_%": g.hit_near.mean() * 100,
        "mean_best_f1": g.best_f1.mean(),
        "mean_rank": g.best_rank.mean(),
        "median_rank": g.best_rank.median(),
        "unique_configs": g.unique_configs.mean(),
        "median_unique_to_near": g.unique_to_near.median(),
    }).reset_index().sort_values(["budget", "hit_near_%", "mean_rank"], ascending=[True, False, True])
    summary.to_csv(out / f"summary_{run_id}.csv", index=False)
    print(f"\nSUMMARY ({len(seeds)} seeds per cell; near = within {args.near} macro-F1 of the best)")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {out / 'runs.csv'} | total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
