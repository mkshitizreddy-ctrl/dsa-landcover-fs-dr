import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mealpy import PSO, FloatVar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dataset_loader import DatasetConfig, load_dataset
from knn import KNN
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
LANDSAT_TRAIN = ROOT / "data" / "sat.trn"
SUMMARY_PATH = ROOT / "results" / "pso_nested_cv_v5.csv"
STABILITY_PATH = ROOT / "results" / "pso_feature_stability_v5.csv"
LOG_PATH = ROOT / "results" / "pso_nested_cv_v5.log"

FIELDS = [
    "run_id", "outer_fold", "method", "seed", "outer_train_n",
    "outer_valid_n", "n_features_selected", "selected_feature_indices",
    "accuracy", "macro_f1", "balanced_accuracy", "inner_best_fitness",
    "objective_evaluations", "failed_objective_evaluations", "epochs",
    "population", "runtime_seconds", "status", "error",
]
STABILITY_FIELDS = ["run_id", "feature_index_zero_based", "selected_folds", "selection_frequency"]


def append_csv(path, fields, row):
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
        writer.writerow(row)


def score(y_true, prediction):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


class MaskObjective:
    def __init__(self, X_train, y_train, X_valid, y_valid, k_neighbors):
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.k_neighbors = k_neighbors
        self.n_features = X_train.shape[1]
        self.best_fitness = float("inf")
        self.best_mask = None
        self.evaluations = 0
        self.failures = 0

    def __call__(self, position):
        self.evaluations += 1
        mask = np.asarray(position, dtype=float) >= 0.5
        if not mask.any():
            mask[int(np.argmax(position))] = True
        try:
            model = KNN(k=self.k_neighbors).fit(self.X_train[:, mask], self.y_train)
            prediction = model.predict(self.X_valid[:, mask])
            macro_f1 = f1_score(self.y_valid, prediction, average="macro", zero_division=0)
            fitness = 0.9 * (1.0 - macro_f1) + 0.1 * (int(mask.sum()) / self.n_features)
            if fitness < self.best_fitness:
                self.best_fitness = float(fitness)
                self.best_mask = mask.copy()
            return float(fitness)
        except Exception:
            self.failures += 1
            return 1.0


def search_pso_mask(split, epochs, population, seed, k_neighbors):
    objective = MaskObjective(split.X_train, split.y_train, split.X_valid, split.y_valid, k_neighbors)
    n = objective.n_features
    problem = {"obj_func": objective, "bounds": FloatVar(lb=[0.0] * n, ub=[1.0] * n),
               "minmax": "min", "log_to": None}
    PSO.OriginalPSO(epoch=epochs, pop_size=population).solve(problem, seed=seed)
    if objective.best_mask is None:
        raise RuntimeError("PSO did not produce a successful candidate")
    return objective


def check_settings(folds, epochs, population, k_neighbors, inner_validation_fraction):
    if folds < 2 or epochs < 1 or population < 2 or k_neighbors < 1:
        raise ValueError("folds >= 2, epochs >= 1, population >= 2 and k_neighbors >= 1 are required")
    if not 0 < inner_validation_fraction < 1:
        raise ValueError("inner_validation_fraction must be between 0 and 1")


def inner_split(dataset, train_idx, fraction, seed):
    inner_dataset = type(dataset)(X=dataset.X[train_idx], y=dataset.y[train_idx],
                                  feature_names=dataset.feature_names, target_name=dataset.target_name,
                                  class_labels=dataset.class_labels, source_path=dataset.source_path)
    return make_validation_split(inner_dataset, validation_fraction=fraction, seed=seed, scale=True)


def outer_rows(dataset, train_idx, valid_idx, selected, search, ctx, fold_started):
    scaler = StandardScaler().fit(dataset.X[train_idx])
    X_train, X_valid = scaler.transform(dataset.X[train_idx]), scaler.transform(dataset.X[valid_idx])
    n_features = dataset.X.shape[1]
    rows = []
    for method, columns in (("all_features", np.arange(n_features)), ("pso_nested", selected)):
        model = KNN(k=ctx["k_neighbors"]).fit(X_train[:, columns], dataset.y[train_idx])
        is_pso = method == "pso_nested"
        rows.append({
            "run_id": ctx["run_id"], "outer_fold": ctx["fold"], "method": method, "seed": ctx["fold_seed"],
            "outer_train_n": len(train_idx), "outer_valid_n": len(valid_idx),
            "n_features_selected": len(columns), "selected_feature_indices": ";".join(map(str, columns.tolist())),
            **score(dataset.y[valid_idx], model.predict(X_valid[:, columns])),
            "inner_best_fitness": search.best_fitness if is_pso else "",
            "objective_evaluations": search.evaluations if is_pso else 0,
            "failed_objective_evaluations": search.failures if is_pso else 0,
            "epochs": ctx["epochs"] if is_pso else "", "population": ctx["population"] if is_pso else "",
            "runtime_seconds": round(time.perf_counter() - fold_started, 3), "status": "success", "error": "",
        })
    return rows


def stability_rows(run_id, n_features, selected_by_fold):
    rows = []
    for feature in range(n_features):
        count = sum(feature in subset for subset in selected_by_fold)
        rows.append({"run_id": run_id, "feature_index_zero_based": feature, "selected_folds": count,
                     "selection_frequency": count / len(selected_by_fold)})
    return rows


def run_pso_nested_cv(data_path=LANDSAT_TRAIN, folds=5, seed=42, inner_validation_fraction=0.2, epochs=5,
                      population=8, k_neighbors=5, save=True, show=print):
    check_settings(folds, epochs, population, k_neighbors, inner_validation_fraction)
    dataset = load_dataset(DatasetConfig(path=str(data_path), delimiter=r"\s+", header=None, target_column=-1))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started_all = time.perf_counter()
    log_lines = [
        f"[{datetime.now(timezone.utc).isoformat()}] START run_id={run_id}",
        f"dataset={data_path} samples={len(dataset.y)} folds={folds} seed={seed}",
        f"PSO epochs={epochs} population={population}; inner validation only; official test not accessed",
    ]
    all_rows, selected_by_fold = [], []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(dataset.X, dataset.y), start=1):
        fold_started = time.perf_counter()
        inner = inner_split(dataset, train_idx, inner_validation_fraction, seed + fold)
        search = search_pso_mask(inner, epochs, population, seed + fold, k_neighbors)
        selected = np.flatnonzero(search.best_mask).astype(int)
        selected_by_fold.append(selected.tolist())
        ctx = {"run_id": run_id, "fold": fold, "fold_seed": seed + fold, "k_neighbors": k_neighbors,
               "epochs": epochs, "population": population}
        rows = outer_rows(dataset, train_idx, valid_idx, selected, search, ctx, fold_started)
        if save:
            for row in rows:
                append_csv(SUMMARY_PATH, FIELDS, row)
        all_rows += rows
        log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] fold={fold}/{folds} "
                         f"features={len(selected)} PSO_macro_f1={rows[1]['macro_f1']:.6f} "
                         f"baseline_macro_f1={rows[0]['macro_f1']:.6f} evaluations={search.evaluations}")
        show(log_lines[-1])

    stability = stability_rows(run_id, dataset.X.shape[1], selected_by_fold)
    log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] END run_id={run_id} "
                     f"elapsed_seconds={time.perf_counter() - started_all:.1f}")
    if save:
        for row in stability:
            append_csv(STABILITY_PATH, STABILITY_FIELDS, row)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(log_lines) + "\n")
    return pd.DataFrame(all_rows, columns=FIELDS), pd.DataFrame(stability, columns=STABILITY_FIELDS)
