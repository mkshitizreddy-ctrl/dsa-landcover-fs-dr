"""Nested-CV comparative study of Mealpy metaheuristics for FS and DR.

Tasks (choose with --tasks):
  fs          wrapper feature selection: search a binary feature mask (meta_fs)
  threshold   tune the heap+BFS correlation threshold t (meta_tune)
  components  tune the PCA component count k (meta_tune)
  joint       tune (t, k) together (meta_tune)

Protocol per outer fold (stratified K-fold over the whole training file):
  1. Inner stratified train/validation split of the outer-training rows only;
     the scaler is fit on inner-train rows (validation_pipeline).
  2. Every algorithm x seed searches with the SAME fitness-evaluation budget.
     Tuning tasks also run an exhaustive grid that gives the true optimum.
  3. The search output is frozen and evaluated on the outer fold:
       fs         - the selected mask is kept as-is;
       t / k      - treated as hyperparameters: MI scores, the correlation
                    graph, heap+BFS and PCA are refit on the outer-training
                    rows with the chosen t and k (k clipped to |S(t)|).
     Scaling and the classifier are fit on outer-training rows only.
The official test file is never read. Outputs are append-only, with a
schema check, in results/mh_comparison_v1.csv and mh_convergence_v1.csv.

New dataset: pass --data, --delimiter, --header, --target; no code changes.
Examples (Windows cmd, run from the repository root):
  python scripts\\run_mh_comparison.py --smoke
  python scripts\\run_mh_comparison.py --data data\\new.csv --delimiter , --header 0 --target label
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_loader import DatasetConfig, load_dataset
from graph import build_corr_graph
from meta_fs import ALGORITHMS, ALGORITHM_FAMILY, MIN_POPULATION, run_feature_selection
from meta_tune import grid_search, run_tuning
from mi import relevance_scores
from pca import PCA
from selection import select_features
from validation_pipeline import make_validation_split

RESULTS = ROOT / "results"
SUMMARY_PATH = RESULTS / "mh_comparison_v1.csv"
CURVE_PATH = RESULTS / "mh_convergence_v1.csv"
LOG_PATH = RESULTS / "mh_comparison_v1.log"

FIELDS = [
    "timestamp_utc", "run_id", "dataset", "task", "method", "family", "seed",
    "outer_fold", "n_features_total", "n_features_used", "n_components",
    "threshold", "accuracy", "macro_f1", "balanced_accuracy",
    "inner_best_fitness", "inner_best_f1", "grid_best_fitness", "hit_grid_optimum",
    "evaluations", "unique_evaluations", "budget", "population", "alpha",
    "search_seconds", "selected_feature_indices", "status", "error",
]
CURVE_FIELDS = ["run_id", "task", "method", "seed", "outer_fold", "epoch", "best_fitness"]
TASKS = ("fs", "threshold", "components", "joint")


# --------------------------------------------------------------------- io
def append_rows(path, fields, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), None)
        if header != fields:
            raise ValueError(f"Refusing to append incompatible schema to {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def parse_header(value):
    return None if value.lower() == "none" else int(value)


def parse_target(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "sat.trn"))
    p.add_argument("--delimiter", default=r"\s+", help=r"e.g. , or \s+")
    p.add_argument("--header", default="none", help="row index of the header, or 'none'")
    p.add_argument("--target", default="-1", help="target column name or integer index")
    p.add_argument("--tasks", default="fs,threshold,components,joint")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="1,2,3", help="search seeds per algorithm and fold")
    p.add_argument("--split-seed", type=int, default=42, help="seed for outer/inner splits")
    p.add_argument("--inner-validation-fraction", type=float, default=0.2)
    p.add_argument("--budget", type=int, default=200, help="fitness evaluations per search")
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="2 folds, 1 seed, 3 algorithms, budget 30: timing check only")
    args = p.parse_args()
    if args.smoke:
        args.folds, args.seeds, args.budget, args.population = 2, "1", 30, 10
        args.algorithms = "PSO,GWO,GA"
    return args


# ------------------------------------------------------------- evaluation
def outer_scores(X_tr, y_tr, X_te, y_te, k_neighbors):
    """Fit kNN on outer-train, score on the outer fold."""
    pred = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_tr, y_tr).predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
    }


def refit_pipeline(X_tr, y_tr, X_te, task, threshold, n_components):
    """Re-run heap+BFS and PCA on outer-train rows with the frozen t / k.

    X_tr, X_te: scaled, Shape [N, D]. Returns transformed (Z_tr, Z_te, S, k).
    """
    D = X_tr.shape[1]
    if task == "components":
        S = list(range(D))
    else:
        C = np.nan_to_num(np.abs(np.corrcoef(X_tr, rowvar=False)), nan=0.0)
        S = sorted(select_features(relevance_scores(X_tr, y_tr), build_corr_graph(C, threshold)))
    if task == "threshold":
        return X_tr[:, S], X_te[:, S], S, len(S)
    k = min(n_components, len(S))
    pca = PCA(max_components=k).fit(X_tr[:, S])
    return pca.transform(X_tr[:, S]), pca.transform(X_te[:, S]), S, k


def base_row(ctx, task, method, seed, fold):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_id": ctx["run_id"],
        "dataset": ctx["dataset"], "task": task, "method": method,
        "family": ALGORITHM_FAMILY.get(method, "baseline" if method == "all_features" else "exhaustive"),
        "seed": seed, "outer_fold": fold, "n_features_total": ctx["D"],
        "n_features_used": "", "n_components": "", "threshold": "",
        "accuracy": "", "macro_f1": "", "balanced_accuracy": "",
        "inner_best_fitness": "", "inner_best_f1": "", "grid_best_fitness": "",
        "hit_grid_optimum": "", "evaluations": "", "unique_evaluations": "",
        "budget": "", "population": "", "alpha": ctx["alpha"], "search_seconds": "",
        "selected_feature_indices": "", "status": "success", "error": "",
    }


def curve_rows(ctx, task, method, seed, fold, curve):
    return [{"run_id": ctx["run_id"], "task": task, "method": method, "seed": seed,
             "outer_fold": fold, "epoch": i + 1, "best_fitness": v} for i, v in enumerate(curve)]


# ------------------------------------------------------------------- main
def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bad = [t for t in tasks if t not in TASKS] + [a for a in algos if a not in ALGORITHMS]
    if bad:
        raise ValueError(f"Unknown task/algorithm: {bad}. Tasks {TASKS}; algorithms {list(ALGORITHMS)}")
    if args.population < MIN_POPULATION or args.budget < args.population or args.folds < 2:
        raise ValueError(f"Need population >= {MIN_POPULATION}, budget >= population, folds >= 2")

    dataset = load_dataset(DatasetConfig(path=args.data, delimiter=args.delimiter,
                                         header=parse_header(args.header),
                                         target_column=parse_target(args.target)))
    X, y = dataset.X, dataset.y                                    # [N, D], [N]
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < args.folds:
        raise ValueError(f"Smallest class has {counts.min()} rows; use --folds <= {counts.min()}")

    ctx = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "dataset": Path(args.data).name, "D": X.shape[1], "alpha": args.alpha}
    log(f"START run_id={ctx['run_id']} data={args.data} N={len(y)} D={X.shape[1]} "
        f"classes={len(counts)} tasks={tasks} algos={algos} folds={args.folds} seeds={seeds} "
        f"budget={args.budget} pop={args.population} alpha={args.alpha} smoke={args.smoke}")
    started = time.perf_counter()

    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        inner_ds = type(dataset)(X=X[tr_idx], y=y[tr_idx], feature_names=dataset.feature_names,
                                 target_name=dataset.target_name, class_labels=dataset.class_labels,
                                 source_path=dataset.source_path)
        inner = make_validation_split(inner_ds, validation_fraction=args.inner_validation_fraction,
                                      seed=args.split_seed + fold, scale=True)
        inner_args = (inner.X_train, inner.y_train, inner.X_valid, inner.y_valid)

        scaler = StandardScaler().fit(X[tr_idx])
        Xo_tr, Xo_te = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
        yo_tr, yo_te = y[tr_idx], y[te_idx]

        rows, curves = [], []
        row = base_row(ctx, "baseline", "all_features", "", fold)
        row.update(n_features_used=ctx["D"], **outer_scores(Xo_tr, yo_tr, Xo_te, yo_te, args.k_neighbors))
        rows.append(row)

        for task in tasks:
            grid = None
            if task != "fs":
                grid = grid_search(*inner_args, mode=task, alpha=args.alpha, k_neighbors=args.k_neighbors)
                Z_tr, Z_te, S, k = refit_pipeline(Xo_tr, yo_tr, Xo_te, task, grid.threshold, grid.n_components)
                row = base_row(ctx, task, "GRID", "", fold)
                row.update(n_features_used=len(S), n_components=k if task != "threshold" else "",
                           threshold=grid.threshold if grid.threshold is not None else "",
                           inner_best_fitness=grid.best_fitness, inner_best_f1=grid.best_inner_f1,
                           grid_best_fitness=grid.best_fitness, hit_grid_optimum=True,
                           evaluations=grid.calls, unique_evaluations=grid.unique_evaluations,
                           search_seconds=grid.runtime_seconds,
                           selected_feature_indices=";".join(map(str, S)),
                           **outer_scores(Z_tr, yo_tr, Z_te, yo_te, args.k_neighbors))
                rows.append(row)

            for name in algos:
                for seed in seeds:
                    row = base_row(ctx, task, name, seed, fold)
                    row.update(budget=args.budget, population=args.population)
                    try:
                        if task == "fs":
                            r = run_feature_selection(name, *inner_args, population=args.population,
                                                      seed=seed, alpha=args.alpha,
                                                      k_neighbors=args.k_neighbors,
                                                      max_evaluations=args.budget)
                            S = r.selected_indices.tolist()
                            scores = outer_scores(Xo_tr[:, S], yo_tr, Xo_te[:, S], yo_te, args.k_neighbors)
                            row.update(n_features_used=len(S), inner_best_fitness=r.best_fitness,
                                       inner_best_f1=r.best_inner_f1, unique_evaluations=r.unique_masks)
                        else:
                            r = run_tuning(name, *inner_args, mode=task, population=args.population,
                                           seed=seed, max_evaluations=args.budget, alpha=args.alpha,
                                           k_neighbors=args.k_neighbors)
                            Z_tr, Z_te, S, k = refit_pipeline(Xo_tr, yo_tr, Xo_te, task,
                                                              r.threshold, r.n_components)
                            scores = outer_scores(Z_tr, yo_tr, Z_te, yo_te, args.k_neighbors)
                            row.update(n_features_used=len(S),
                                       n_components=k if task != "threshold" else "",
                                       threshold=r.threshold if r.threshold is not None else "",
                                       inner_best_fitness=r.best_fitness, inner_best_f1=r.best_inner_f1,
                                       grid_best_fitness=grid.best_fitness,
                                       hit_grid_optimum=bool(abs(r.best_fitness - grid.best_fitness) < 1e-12),
                                       unique_evaluations=r.unique_evaluations)
                        row.update(evaluations=r.calls, search_seconds=r.runtime_seconds,
                                   selected_feature_indices=";".join(map(str, S)), **scores)
                        curves += curve_rows(ctx, task, name, seed, fold, r.convergence)
                    except Exception as exc:  # keep the study running; record the failure
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
                    rows.append(row)

        append_rows(SUMMARY_PATH, FIELDS, rows)
        append_rows(CURVE_PATH, CURVE_FIELDS, curves)
        ok = [r for r in rows if r["status"] == "success" and r["method"] not in ("all_features", "GRID")]
        failed = sum(r["status"] == "failed" for r in rows)
        log(f"fold {fold}/{args.folds} done: {len(rows)} rows, {failed} failed, "
            f"all_features macro_f1={rows[0]['macro_f1']:.4f}, "
            f"mean search macro_f1={np.mean([r['macro_f1'] for r in ok]):.4f}, "
            f"elapsed {time.perf_counter() - started:.1f}s")

    log(f"END run_id={ctx['run_id']} elapsed_seconds={time.perf_counter() - started:.1f} "
        f"-> {SUMMARY_PATH.name}, {CURVE_PATH.name}")


if __name__ == "__main__":
    main()
