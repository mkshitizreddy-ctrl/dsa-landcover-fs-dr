"""Freeze PSO settings on sat.trn, then evaluate once on the official test set.

The official test file is intentionally loaded only after the selected mask
and all hyperparameters have been fixed using a stratified split of sat.trn.
This is a separate final evaluation stage, not a tuning script.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mealpy import FloatVar, PSO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from dataset_loader import DatasetConfig, load_dataset
from knn import KNN
from validation_pipeline import make_validation_split

OUT_PATH = ROOT / "results" / "pso_official_test_v1.csv"
LOG_PATH = ROOT / "results" / "pso_official_test_v1.log"
FIELDS = [
    "run_id", "evaluation_stage", "method", "training_data", "test_data",
    "seed", "n_train", "n_test", "n_features_total", "n_features_selected",
    "selected_feature_indices", "k_neighbors", "validation_fraction",
    "epochs", "population", "objective_evaluations", "failed_objective_evaluations",
    "accuracy", "macro_f1", "balanced_accuracy", "runtime_seconds", "status",
]


def append_row(row):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT_PATH.exists() and OUT_PATH.stat().st_size > 0
    if exists:
        with OUT_PATH.open(newline="", encoding="utf-8-sig") as stream:
            if next(csv.reader(stream), None) != FIELDS:
                raise ValueError(f"Refusing to append incompatible schema to {OUT_PATH}")
    with OUT_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--k-neighbors", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        raise FileExistsError(
            f"Final test output already exists; refusing a second official-test evaluation: {OUT_PATH}"
        )
    if args.epochs < 1 or args.population < 2 or args.k_neighbors < 1:
        raise ValueError("epochs >= 1, population >= 2 and k-neighbors >= 1 are required")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    train_path = ROOT / "data" / "sat.trn"
    train = load_dataset(DatasetConfig(path=str(train_path), delimiter=r"\s+", header=None, target_column=-1))
    split = make_validation_split(
        train, validation_fraction=args.validation_fraction, seed=args.seed, scale=True,
    )
    n_features = split.X_train.shape[1]
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
            model = KNN(k=args.k_neighbors).fit(split.X_train[:, mask], split.y_train)
            prediction = model.predict(split.X_valid[:, mask])
            macro_f1 = f1_score(split.y_valid, prediction, average="macro", zero_division=0)
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
    started = time.perf_counter()
    optimizer = PSO.OriginalPSO(epoch=args.epochs, pop_size=args.population)
    optimizer.solve(problem, seed=args.seed)
    if best_mask is None:
        raise RuntimeError("PSO did not produce a successful training-only candidate")
    selected = np.flatnonzero(best_mask).astype(int)

    # Freeze the chosen mask and preprocessing before opening the official test data.
    full_scaler = split.scaler.__class__().fit(train.X)
    X_train = full_scaler.transform(train.X)
    selected_train_model = KNN(k=args.k_neighbors).fit(X_train[:, selected], train.y)
    all_train_model = KNN(k=args.k_neighbors).fit(X_train, train.y)
    freeze_line = (
        f"[{datetime.now(timezone.utc).isoformat()}] FROZEN run_id={run_id} "
        f"seed={args.seed} epochs={args.epochs} population={args.population} "
        f"k={args.k_neighbors} validation_fraction={args.validation_fraction} "
        f"features={selected.tolist()} fitness={best_fitness:.8f}; now loading official test"
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] START run_id={run_id}\n")
        stream.write(f"Training-only search: evaluations={evaluations}, failures={failures}\n")
        stream.write(freeze_line + "\n")

    # A header-only file acts as a one-shot guard even if the test stage fails.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    test_path = ROOT / "data" / "sat.tst"
    test = load_dataset(DatasetConfig(path=str(test_path), delimiter=r"\s+", header=None, target_column=-1))
    X_test = full_scaler.transform(test.X)
    results = []
    for method, model, columns in (
        ("all_features", all_train_model, np.arange(n_features)),
        ("pso_selected", selected_train_model, selected),
    ):
        prediction = model.predict(X_test[:, columns])
        results.append({
            "run_id": run_id, "evaluation_stage": "single_final_official_test",
            "method": method, "training_data": str(train_path), "test_data": str(test_path),
            "seed": args.seed, "n_train": len(train.y), "n_test": len(test.y),
            "n_features_total": n_features, "n_features_selected": len(columns),
            "selected_feature_indices": ";".join(map(str, columns.tolist())),
            "k_neighbors": args.k_neighbors, "validation_fraction": args.validation_fraction,
            "epochs": args.epochs if method == "pso_selected" else "",
            "population": args.population if method == "pso_selected" else "",
            "objective_evaluations": evaluations if method == "pso_selected" else 0,
            "failed_objective_evaluations": failures if method == "pso_selected" else 0,
            "accuracy": float(accuracy_score(test.y, prediction)),
            "macro_f1": float(f1_score(test.y, prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(test.y, prediction)),
            "runtime_seconds": round(time.perf_counter() - started, 3), "status": "success",
        })
    for row in results:
        append_row(row)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now(timezone.utc).isoformat()}] FINAL rows_written={len(results)} output={OUT_PATH}\n")
    print(f"Frozen feature indices (zero-based): {selected.tolist()}")
    for row in results:
        print(f"{row['method']}: accuracy={row['accuracy']:.6f}, macro-F1={row['macro_f1']:.6f}, balanced accuracy={row['balanced_accuracy']:.6f}")
    print(f"Saved final official test results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
