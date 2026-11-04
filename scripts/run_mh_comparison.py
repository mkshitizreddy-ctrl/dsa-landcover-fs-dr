import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dataset_loader import DatasetConfig, load_dataset
from graph import build_corr_graph
from meta_fs import ALGORITHM_FAMILY, ALGORITHMS, MIN_POPULATION, run_feature_selection
from meta_tune import grid_search, run_tuning
from mi import relevance_scores
from pca import PCA
from selection import select_features
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY_PATH = RESULTS / "mh_comparison_v1.csv"
CURVE_PATH = RESULTS / "mh_convergence_v1.csv"
LOG_PATH = RESULTS / "mh_comparison_v1.log"
LANDSAT_TRAIN = ROOT / "data" / "sat.trn"
TASKS = ("fs", "threshold", "components", "joint")
QUICK_CHECK = {"folds": 2, "seeds": (1,), "budget": 30, "population": 10, "algorithms": ("PSO", "GWO", "GA")}

FIELDS = [
    "timestamp_utc", "run_id", "dataset", "task", "method", "family", "seed",
    "outer_fold", "n_features_total", "n_features_used", "n_components",
    "threshold", "accuracy", "macro_f1", "balanced_accuracy",
    "inner_best_fitness", "inner_best_f1", "grid_best_fitness", "hit_grid_optimum",
    "evaluations", "unique_evaluations", "budget", "population", "alpha",
    "search_seconds", "selected_feature_indices", "status", "error",
]
CURVE_FIELDS = ["run_id", "task", "method", "seed", "outer_fold", "epoch", "best_fitness"]


def append_rows(path, fields, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError(f"Refusing to append incompatible schema to {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def make_logger(show, save):
    def log(message):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        show(line)
        if save:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    return log


def check_settings(tasks, algorithms, population, budget, folds, y):
    unknown = [t for t in tasks if t not in TASKS] + [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Unknown task/algorithm: {unknown}. Tasks {TASKS}; algorithms {list(ALGORITHMS)}")
    if population < MIN_POPULATION or budget < population or folds < 2:
        raise ValueError(f"Need population >= {MIN_POPULATION}, budget >= population, folds >= 2")
    smallest = np.unique(y, return_counts=True)[1].min()
    if smallest < folds:
        raise ValueError(f"Smallest class has {smallest} rows; use at most {smallest} folds")


def outer_scores(X_tr, y_tr, X_te, y_te, k_neighbors):
    pred = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_tr, y_tr).predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
    }


def refit_pipeline(X_tr, y_tr, X_te, task, threshold, n_components):
    if task == "components":
        S = list(range(X_tr.shape[1]))
    else:
        C = np.nan_to_num(np.abs(np.corrcoef(X_tr, rowvar=False)), nan=0.0)
        S = sorted(select_features(relevance_scores(X_tr, y_tr), build_corr_graph(C, threshold)))
    if task == "threshold":
        return X_tr[:, S], X_te[:, S], S, len(S)
    k = min(n_components, len(S))
    pca = PCA(max_components=k).fit(X_tr[:, S])
    return pca.transform(X_tr[:, S]), pca.transform(X_te[:, S]), S, k


def base_row(ctx, task, method, seed, fold):
    family = ALGORITHM_FAMILY.get(method, "baseline" if method == "all_features" else "exhaustive")
    row = {f: "" for f in FIELDS}
    row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), run_id=ctx["run_id"],
               dataset=ctx["dataset"], task=task, method=method, family=family, seed=seed,
               outer_fold=fold, n_features_total=ctx["D"], alpha=ctx["alpha"], status="success")
    return row


def curve_rows(ctx, task, method, seed, fold, curve):
    return [{"run_id": ctx["run_id"], "task": task, "method": method, "seed": seed,
             "outer_fold": fold, "epoch": i + 1, "best_fitness": v} for i, v in enumerate(curve)]


def tuning_details(task, S, k, result):
    return {"n_features_used": len(S),
            "n_components": k if task != "threshold" else "",
            "threshold": result.threshold if result.threshold is not None else "",
            "inner_best_fitness": result.best_fitness, "inner_best_f1": result.best_inner_f1,
            "unique_evaluations": result.unique_evaluations}


def grid_row(ctx, task, fold, inner, outer, settings):
    grid = grid_search(*inner, mode=task, alpha=settings["alpha"], k_neighbors=settings["k_neighbors"])
    Xo_tr, yo_tr, Xo_te, yo_te = outer
    Z_tr, Z_te, S, k = refit_pipeline(Xo_tr, yo_tr, Xo_te, task, grid.threshold, grid.n_components)
    row = base_row(ctx, task, "GRID", "", fold)
    row.update(**tuning_details(task, S, k, grid), grid_best_fitness=grid.best_fitness, hit_grid_optimum=True,
               evaluations=grid.calls, search_seconds=grid.runtime_seconds,
               selected_feature_indices=";".join(map(str, S)),
               **outer_scores(Z_tr, yo_tr, Z_te, yo_te, settings["k_neighbors"]))
    return row, grid


def search_row(ctx, task, name, seed, fold, inner, outer, grid, settings):
    Xo_tr, yo_tr, Xo_te, yo_te = outer
    common = {"population": settings["population"], "seed": seed, "max_evaluations": settings["budget"],
              "alpha": settings["alpha"], "k_neighbors": settings["k_neighbors"]}
    row = base_row(ctx, task, name, seed, fold)
    row.update(budget=settings["budget"], population=settings["population"])
    try:
        if task == "fs":
            result = run_feature_selection(name, *inner, **common)
            S = result.selected_indices.tolist()
            scores = outer_scores(Xo_tr[:, S], yo_tr, Xo_te[:, S], yo_te, settings["k_neighbors"])
            row.update(n_features_used=len(S), inner_best_fitness=result.best_fitness,
                       inner_best_f1=result.best_inner_f1, unique_evaluations=result.unique_masks)
        else:
            result = run_tuning(name, *inner, mode=task, **common)
            Z_tr, Z_te, S, k = refit_pipeline(Xo_tr, yo_tr, Xo_te, task, result.threshold, result.n_components)
            scores = outer_scores(Z_tr, yo_tr, Z_te, yo_te, settings["k_neighbors"])
            row.update(**tuning_details(task, S, k, result), grid_best_fitness=grid.best_fitness,
                       hit_grid_optimum=bool(abs(result.best_fitness - grid.best_fitness) < 1e-12))
        row.update(evaluations=result.calls, search_seconds=result.runtime_seconds,
                   selected_feature_indices=";".join(map(str, S)), **scores)
        return row, curve_rows(ctx, task, name, seed, fold, result.convergence)
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
        return row, []


def fold_data(dataset, train_idx, test_idx, fold, settings):
    X, y = dataset.X, dataset.y
    inner_ds = type(dataset)(X=X[train_idx], y=y[train_idx], feature_names=dataset.feature_names,
                             target_name=dataset.target_name, class_labels=dataset.class_labels,
                             source_path=dataset.source_path)
    split = make_validation_split(inner_ds, validation_fraction=settings["inner_validation_fraction"],
                                  seed=settings["split_seed"] + fold, scale=True)
    inner = (split.X_train, split.y_train, split.X_valid, split.y_valid)
    scaler = StandardScaler().fit(X[train_idx])
    outer = (scaler.transform(X[train_idx]), y[train_idx], scaler.transform(X[test_idx]), y[test_idx])
    return inner, outer


def run_fold(ctx, fold, inner, outer, tasks, algorithms, seeds, settings):
    rows, curves = [], []
    baseline = base_row(ctx, "baseline", "all_features", "", fold)
    baseline.update(n_features_used=ctx["D"], **outer_scores(*outer, settings["k_neighbors"]))
    rows.append(baseline)
    for task in tasks:
        grid = None
        if task != "fs":
            row, grid = grid_row(ctx, task, fold, inner, outer, settings)
            rows.append(row)
        for name in algorithms:
            for seed in seeds:
                row, curve = search_row(ctx, task, name, seed, fold, inner, outer, grid, settings)
                rows.append(row)
                curves += curve
    return rows, curves


def fold_message(fold, folds, rows, elapsed):
    searched = [r for r in rows if r["status"] == "success" and r["method"] not in ("all_features", "GRID")]
    failed = sum(r["status"] == "failed" for r in rows)
    return (f"fold {fold}/{folds} done: {len(rows)} rows, {failed} failed, "
            f"all_features macro_f1={rows[0]['macro_f1']:.4f}, "
            f"mean search macro_f1={np.mean([r['macro_f1'] for r in searched]):.4f}, elapsed {elapsed:.1f}s")


def run_mh_comparison(data_path=LANDSAT_TRAIN, delimiter=r"\s+", header=None, target=-1, tasks=TASKS,
                      algorithms=tuple(ALGORITHMS), folds=5, seeds=(1, 2, 3), split_seed=42,
                      inner_validation_fraction=0.2, budget=200, population=20, alpha=0.9, k_neighbors=5,
                      quick_check=False, save=True, show=print):
    if quick_check:
        folds, seeds, budget = QUICK_CHECK["folds"], QUICK_CHECK["seeds"], QUICK_CHECK["budget"]
        population, algorithms = QUICK_CHECK["population"], QUICK_CHECK["algorithms"]
    dataset = load_dataset(DatasetConfig(path=str(data_path), delimiter=delimiter, header=header,
                                         target_column=target))
    check_settings(tasks, algorithms, population, budget, folds, dataset.y)
    settings = {"budget": budget, "population": population, "alpha": alpha, "k_neighbors": k_neighbors,
                "split_seed": split_seed, "inner_validation_fraction": inner_validation_fraction}
    ctx = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "dataset": Path(data_path).name, "D": dataset.X.shape[1], "alpha": alpha}
    log = make_logger(show, save)
    log(f"START run_id={ctx['run_id']} data={data_path} N={len(dataset.y)} D={ctx['D']} "
        f"classes={len(np.unique(dataset.y))} tasks={list(tasks)} algos={list(algorithms)} folds={folds} "
        f"seeds={list(seeds)} budget={budget} pop={population} alpha={alpha} quick_check={quick_check}")
    started = time.perf_counter()

    all_rows, all_curves = [], []
    outer_split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    for fold, (train_idx, test_idx) in enumerate(outer_split.split(dataset.X, dataset.y), start=1):
        inner, outer = fold_data(dataset, train_idx, test_idx, fold, settings)
        rows, curves = run_fold(ctx, fold, inner, outer, tasks, algorithms, seeds, settings)
        if save:
            append_rows(SUMMARY_PATH, FIELDS, rows)
            append_rows(CURVE_PATH, CURVE_FIELDS, curves)
        all_rows += rows
        all_curves += curves
        log(fold_message(fold, folds, rows, time.perf_counter() - started))

    log(f"END run_id={ctx['run_id']} elapsed_seconds={time.perf_counter() - started:.1f}")
    return pd.DataFrame(all_rows, columns=FIELDS), pd.DataFrame(all_curves, columns=CURVE_FIELDS)
