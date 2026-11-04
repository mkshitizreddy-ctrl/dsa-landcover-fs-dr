import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dataset_loader import DatasetConfig, load_dataset
from knn import KNN
from run_pso_nested_cv_v5 import search_pso_mask
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "sat.trn"
TEST_PATH = ROOT / "data" / "sat.tst"
OUT_PATH = ROOT / "results" / "pso_official_test_v1.csv"
LOG_PATH = ROOT / "results" / "pso_official_test_v1.log"
FIELDS = [
    "run_id", "evaluation_stage", "method", "training_data", "test_data",
    "seed", "n_train", "n_test", "n_features_total", "n_features_selected",
    "selected_feature_indices", "k_neighbors", "validation_fraction",
    "epochs", "population", "objective_evaluations", "failed_objective_evaluations",
    "accuracy", "macro_f1", "balanced_accuracy", "runtime_seconds", "status",
]


def load_landsat_file(path):
    return load_dataset(DatasetConfig(path=str(path), delimiter=r"\s+", header=None, target_column=-1))


def official_test_used():
    return OUT_PATH.exists() and OUT_PATH.stat().st_size > 0


def saved_results():
    return pd.read_csv(OUT_PATH) if official_test_used() else pd.DataFrame(columns=FIELDS)


def check_settings(epochs, population, k_neighbors, validation_fraction):
    if official_test_used():
        raise FileExistsError(f"Final test output already exists; refusing a second official-test "
                              f"evaluation: {OUT_PATH}")
    if epochs < 1 or population < 2 or k_neighbors < 1:
        raise ValueError("epochs >= 1, population >= 2 and k_neighbors >= 1 are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")


def write_log(lines):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def lock_official_test():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writeheader()


def append_rows(rows):
    with OUT_PATH.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writerows(rows)


def test_row(method, columns, prediction, test, ctx):
    is_pso = method == "pso_selected"
    return {
        "run_id": ctx["run_id"], "evaluation_stage": "single_final_official_test", "method": method,
        "training_data": str(TRAIN_PATH), "test_data": str(TEST_PATH), "seed": ctx["seed"],
        "n_train": ctx["n_train"], "n_test": len(test.y), "n_features_total": ctx["n_features"],
        "n_features_selected": len(columns), "selected_feature_indices": ";".join(map(str, columns.tolist())),
        "k_neighbors": ctx["k_neighbors"], "validation_fraction": ctx["validation_fraction"],
        "epochs": ctx["epochs"] if is_pso else "", "population": ctx["population"] if is_pso else "",
        "objective_evaluations": ctx["evaluations"] if is_pso else 0,
        "failed_objective_evaluations": ctx["failures"] if is_pso else 0,
        "accuracy": float(accuracy_score(test.y, prediction)),
        "macro_f1": float(f1_score(test.y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(test.y, prediction)),
        "runtime_seconds": round(time.perf_counter() - ctx["started"], 3), "status": "success",
    }


def run_pso_final_test(seed=42, validation_fraction=0.2, epochs=10, population=10, k_neighbors=5, show=print):
    check_settings(epochs, population, k_neighbors, validation_fraction)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    train = load_landsat_file(TRAIN_PATH)
    split = make_validation_split(train, validation_fraction=validation_fraction, seed=seed, scale=True)
    started = time.perf_counter()
    search = search_pso_mask(split, epochs, population, seed, k_neighbors)
    selected = np.flatnonzero(search.best_mask).astype(int)
    n_features = split.X_train.shape[1]

    full_scaler = split.scaler.__class__().fit(train.X)
    X_train = full_scaler.transform(train.X)
    models = {"all_features": (KNN(k=k_neighbors).fit(X_train, train.y), np.arange(n_features)),
              "pso_selected": (KNN(k=k_neighbors).fit(X_train[:, selected], train.y), selected)}
    now = datetime.now(timezone.utc).isoformat()
    write_log([f"\n[{now}] START run_id={run_id}",
               f"Training-only search: evaluations={search.evaluations}, failures={search.failures}",
               f"[{now}] FROZEN run_id={run_id} seed={seed} epochs={epochs} population={population} "
               f"k={k_neighbors} validation_fraction={validation_fraction} features={selected.tolist()} "
               f"fitness={search.best_fitness:.8f}; now loading official test"])
    lock_official_test()

    test = load_landsat_file(TEST_PATH)
    X_test = full_scaler.transform(test.X)
    ctx = {"run_id": run_id, "seed": seed, "n_train": len(train.y), "n_features": n_features,
           "k_neighbors": k_neighbors, "validation_fraction": validation_fraction, "epochs": epochs,
           "population": population, "evaluations": search.evaluations, "failures": search.failures,
           "started": started}
    rows = [test_row(method, columns, model.predict(X_test[:, columns]), test, ctx)
            for method, (model, columns) in models.items()]
    append_rows(rows)
    write_log([f"[{datetime.now(timezone.utc).isoformat()}] FINAL rows_written={len(rows)} output={OUT_PATH}"])
    show(f"Frozen feature indices (zero-based): {selected.tolist()}")
    return pd.DataFrame(rows, columns=FIELDS)
