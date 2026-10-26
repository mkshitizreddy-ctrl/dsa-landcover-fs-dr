"""Nested stratified CV for a leakage-safe PSO feature-selection estimate.

Each outer fold is held out from both PSO search and preprocessing. PSO uses
an inner train/validation split from the outer-training rows only. Outputs are
new v5 files; historical V1-V4 artifacts are never edited.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mealpy import FloatVar, PSO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from dataset_loader import DatasetConfig, load_dataset
from knn import KNN
from validation_pipeline import make_validation_split

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
            header = next(csv.reader(stream), None)
        if header != fields:
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--k-neighbors", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.population < 2 or args.k_neighbors < 1:
        raise ValueError("folds >= 2, epochs >= 1, population >= 2 and k-neighbors >= 1 are required")
    if not 0 < args.inner_validation_fraction < 1:
        raise ValueError("inner-validation-fraction must be between 0 and 1")

    data_path = ROOT / "data" / "sat.trn"
    dataset = load_dataset(DatasetConfig(path=str(data_path), delimiter=r"\s+", header=None, target_column=-1))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    selected_by_fold = []
    started_all = time.perf_counter()
    log_lines = [
        f"[{datetime.now(timezone.utc).isoformat()}] START run_id={run_id}",
        f"dataset={data_path} samples={len(dataset.y)} folds={args.folds} seed={args.seed}",
        f"PSO epochs={args.epochs} population={args.population}; inner validation only; official test not accessed",
    ]

    for fold, (outer_train_idx, outer_valid_idx) in enumerate(splitter.split(dataset.X, dataset.y), start=1):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        inner_dataset = type(dataset)(
            X=dataset.X[outer_train_idx], y=dataset.y[outer_train_idx],
            feature_names=dataset.feature_names, target_name=dataset.target_name,
            class_labels=dataset.class_labels, source_path=dataset.source_path,
        )
        inner = make_validation_split(
            inner_dataset, validation_fraction=args.inner_validation_fraction,
            seed=fold_seed, scale=True,
        )
        n_features = inner.X_train.shape[1]
        best_fitness = float("inf")
        best_mask = None
        evaluations = 0
        failures = 0

        def objective(position):
            nonlocal best_fitness, best_mask, evaluations, failures
            evaluations += 1
            mask = np.asarray(position, dtype=float) >= 0.5
            if not mask.any():
                mask[int(np.argmax(position))] = True
            try:
                model = KNN(k=args.k_neighbors).fit(inner.X_train[:, mask], inner.y_train)
                pred = model.predict(inner.X_valid[:, mask])
                macro_f1 = f1_score(inner.y_valid, pred, average="macro", zero_division=0)
                fitness = 0.9 * (1.0 - macro_f1) + 0.1 * (int(mask.sum()) / n_features)
                if fitness < best_fitness:
                    best_fitness = float(fitness)
                    best_mask = mask.copy()
                return float(fitness)
            except Exception:
                failures += 1
                return 1.0

        problem = {
            "obj_func": objective,
            "bounds": FloatVar(lb=[0.0] * n_features, ub=[1.0] * n_features),
            "minmax": "min",
            "log_to": None,
        }
        optimizer = PSO.OriginalPSO(epoch=args.epochs, pop_size=args.population)
        optimizer.solve(problem, seed=fold_seed)
        if best_mask is None:
            raise RuntimeError(f"No successful PSO candidate in outer fold {fold}")
        selected = np.flatnonzero(best_mask).astype(int)
        selected_by_fold.append(selected.tolist())

        # Refit preprocessing and classifier on every outer-training row only.
        scaler = StandardScaler().fit(dataset.X[outer_train_idx])
        X_outer_train = scaler.transform(dataset.X[outer_train_idx])
        X_outer_valid = scaler.transform(dataset.X[outer_valid_idx])
        rows = []
        for method, columns in (("all_features", np.arange(n_features)), ("pso_nested", selected)):
            model = KNN(k=args.k_neighbors).fit(X_outer_train[:, columns], dataset.y[outer_train_idx])
            prediction = model.predict(X_outer_valid[:, columns])
            metrics = score(dataset.y[outer_valid_idx], prediction)
            rows.append({
                "run_id": run_id, "outer_fold": fold, "method": method,
                "seed": fold_seed, "outer_train_n": len(outer_train_idx),
                "outer_valid_n": len(outer_valid_idx),
                "n_features_selected": len(columns),
                "selected_feature_indices": ";".join(map(str, columns.tolist())),
                **metrics, "inner_best_fitness": best_fitness if method == "pso_nested" else "",
                "objective_evaluations": evaluations if method == "pso_nested" else 0,
                "failed_objective_evaluations": failures if method == "pso_nested" else 0,
                "epochs": args.epochs if method == "pso_nested" else "",
                "population": args.population if method == "pso_nested" else "",
                "runtime_seconds": round(time.perf_counter() - fold_started, 3),
                "status": "success", "error": "",
            })
        for row in rows:
            append_csv(SUMMARY_PATH, FIELDS, row)
        log_lines.append(
            f"[{datetime.now(timezone.utc).isoformat()}] fold={fold}/{args.folds} "
            f"features={len(selected)} PSO_macro_f1={rows[1]['macro_f1']:.6f} "
            f"baseline_macro_f1={rows[0]['macro_f1']:.6f} evaluations={evaluations}"
        )
        print(log_lines[-1])

    for feature in range(dataset.X.shape[1]):
        count = sum(feature in subset for subset in selected_by_fold)
        append_csv(STABILITY_PATH, STABILITY_FIELDS, {
            "run_id": run_id,
            "feature_index_zero_based": feature,
            "selected_folds": count,
            "selection_frequency": count / args.folds,
        })
    log_lines.append(
        f"[{datetime.now(timezone.utc).isoformat()}] END run_id={run_id} "
        f"elapsed_seconds={time.perf_counter() - started_all:.1f}"
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(log_lines) + "\n")
    print(f"\nSaved nested CV rows: {SUMMARY_PATH}")
    print(f"Saved feature stability: {STABILITY_PATH}")


if __name__ == "__main__":
    main()
