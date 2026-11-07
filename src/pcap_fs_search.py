import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mealpy import FloatVar

from meta_fs import ALGORITHM_FAMILY, ALGORITHMS, decode_mask, make_optimizer_and_stop
from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_feature_selection import FS_DIR, METHODS, THRESHOLDS, prepare_fold
from pcap_grid import CLASSIFIERS, SCALERS, check_choices, evaluate
from pcap_mh_search import (RANDOM, GridSpace, append_rows, check_algorithms, p_random_hit, pick_option,
                            run_algorithm, run_random, summarize_runs)

ROOT = Path(__file__).resolve().parents[1]
FS_SUMMARY_PATH = FS_DIR / "fs_summary.csv"
FS_SEARCH_DIR = ROOT / "results" / "pcap_fs_search_v1"
MASK_DIR = ROOT / "results" / "pcap_mask_search_v1"
MASK_QUICK_CHECK_DIR = ROOT / "results" / "pcap_mask_search_quick_check"
MASK_QUICK_CHECK = {"algorithms": ("PSO", "GWO", RANDOM), "seeds": (1,), "budget": 30, "population": 10}


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


class MaskObjective:

    def __init__(self, fold_data, class_ids, classifier="DT", alpha=0.9, threshold=0.5, model_seed=42):
        self.fold_data, self.class_ids, self.classifier = fold_data, class_ids, classifier
        self.alpha, self.threshold, self.model_seed = alpha, threshold, model_seed
        self.n_features = fold_data[0][0].shape[1]
        self.cache = {}
        self.calls = 0
        self.best_fitness, self.best_f1, self.best_cols = math.inf, 0.0, None
        self.curve = []

    def cv_f1(self, cols):
        scores = [evaluate(self.classifier, Z_tr[:, cols], y_tr, Z_va[:, cols], y_va, self.class_ids,
                           self.model_seed).get("f1_macro", 0.0) for Z_tr, y_tr, Z_va, y_va in self.fold_data]
        return float(np.mean(scores))

    def __call__(self, position):
        self.calls += 1
        cols = tuple(int(j) for j in np.flatnonzero(decode_mask(position, self.threshold)))
        if cols not in self.cache:
            f1 = self.cv_f1(list(cols))
            fitness = self.alpha * (1.0 - f1) + (1.0 - self.alpha) * len(cols) / self.n_features
            self.cache[cols] = fitness
            if fitness < self.best_fitness:
                self.best_fitness, self.best_f1, self.best_cols = fitness, f1, cols
            self.curve.append(self.best_f1)
        return self.cache[cols]


def search_mask(name, fold_data, class_ids, budget, population, seed, classifier="DT", alpha=0.9):
    objective = MaskObjective(fold_data, class_ids, classifier, alpha)
    start = time.perf_counter()
    if name == RANDOM:
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            objective(rng.random(objective.n_features))
    else:
        n = objective.n_features
        problem = {"obj_func": objective, "bounds": FloatVar(lb=[0.0] * n, ub=[1.0] * n),
                   "minmax": "min", "log_to": None}
        optimizer, termination = make_optimizer_and_stop(name, 1, population, budget)
        optimizer.solve(problem, seed=seed, termination=termination)
    objective.fold_data = None
    return objective, time.perf_counter() - start


def mask_row(name, seed, objective, seconds, feature_names, settings):
    cols = objective.best_cols
    return {"run_id": settings["run_id"], "algorithm": name, "family": ALGORITHM_FAMILY.get(name, "baseline"),
            "seed": seed, "budget": settings["budget"], "population": settings["population"],
            "scaler": settings["scaler"], "classifier": settings["classifier"], "alpha": settings["alpha"],
            "calls": objective.calls, "unique_masks": len(objective.cache), "best_fitness": objective.best_fitness,
            "cv_f1_macro": objective.best_f1, "n_features": len(cols),
            "features": ";".join(feature_names[j] for j in cols), "seconds": round(seconds, 1)}


def curve_rows(name, seed, objective, run_id):
    return [{"run_id": run_id, "algorithm": name, "seed": seed, "masks_tried": i, "best_cv_f1_macro": f1}
            for i, f1 in enumerate(objective.curve, start=1)]


def summarize_masks(runs):
    groups = pd.DataFrame(runs).groupby("algorithm")
    summary = pd.DataFrame({
        "mean_cv_f1_macro": groups.cv_f1_macro.mean(),
        "best_cv_f1_macro": groups.cv_f1_macro.max(),
        "mean_n_features": groups.n_features.mean(),
        "mean_unique_masks": groups.unique_masks.mean(),
        "mean_seconds": groups.seconds.mean(),
    }).reset_index()
    return summary.sort_values("mean_cv_f1_macro", ascending=False, kind="stable")


def load_fold_data(data_path, splits_path, scaler):
    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X, y = data.X[splits.features].to_numpy(dtype=float), data.y
    fold_data = []
    for tr, va in splits.folds:
        Z_train, Z_valid = prepare_fold(X, tr, va, scaler)
        fold_data.append((Z_train, y[tr], Z_valid, y[va]))
    return fold_data, list(range(len(data.class_names))), list(splits.features)


def run_mask_search(data_path=DATA_PATH, splits_path=SPLITS_PATH, algorithms=tuple(ALGORITHMS) + (RANDOM,),
                    seeds=(1, 2, 3), budget=200, population=20, scaler="Normalizer", classifier="DT", alpha=0.9,
                    jobs=None, quick_check=False, save=True, out_dir=None, show=print):
    if quick_check:
        algorithms, seeds = MASK_QUICK_CHECK["algorithms"], MASK_QUICK_CHECK["seeds"]
        budget, population = MASK_QUICK_CHECK["budget"], MASK_QUICK_CHECK["population"]
    check_algorithms(algorithms)
    check_choices((scaler,), (), (classifier,))
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    out_dir = Path(out_dir or (MASK_QUICK_CHECK_DIR if quick_check else MASK_DIR))
    fold_data, class_ids, feature_names = load_fold_data(data_path, splits_path, scaler)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = {"run_id": run_id, "budget": budget, "population": population, "scaler": scaler,
                "classifier": classifier, "alpha": alpha}
    tasks = [(name, seed) for name in algorithms for seed in seeds]
    show(f"Mask search: {len(tasks)} runs ({len(algorithms)} algorithms x {len(seeds)} seeds), budget {budget}, "
         f"{scaler} + {classifier}, {len(fold_data)} folds, {jobs} parallel jobs")

    started, runs, curves = time.perf_counter(), [], []
    for i in range(0, len(tasks), jobs):
        chunk = tasks[i:i + jobs]
        results = Parallel(n_jobs=min(jobs, len(chunk)))(
            delayed(search_mask)(name, fold_data, class_ids, budget, population, seed, classifier, alpha)
            for name, seed in chunk)
        for (name, seed), (objective, seconds) in zip(chunk, results):
            runs.append(mask_row(name, seed, objective, seconds, feature_names, settings))
            curves += curve_rows(name, seed, objective, run_id)
            show(f"{name:6} seed {seed}: CV macro-F1 {objective.best_f1:.4f} with {len(objective.best_cols)} "
                 f"features, {len(objective.cache)} masks | elapsed {time.perf_counter() - started:.0f}s")

    summary = summarize_masks(runs)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        append_rows(out_dir / "runs.csv", runs)
        append_rows(out_dir / "curves.csv", curves)
        summary.to_csv(out_dir / f"summary_{run_id}.csv", index=False)
        show(f"Saved {out_dir / 'runs.csv'}")
    return pd.DataFrame(runs), summary
