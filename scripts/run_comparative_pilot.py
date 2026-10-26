
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from mealpy import PSO, FloatVar
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sat.trn"
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))

from knn import KNN
from mi import relevance_scores
from graph import build_corr_graph
from selection import select_features

SEED = 42
VALIDATION_SIZE = 0.20
KNN_K = 5

# Small pilot budget. Expand only after validating this pipeline.
PSO_EPOCHS = 5
PSO_POPULATION = 8

SUMMARY = RESULTS / "comparative_pilot.csv"
LOG = RESULTS / "comparative_pilot.log"
ERROR_LOG = RESULTS / "comparative_pilot_errors.log"


def log(message):
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(name):
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] {name}\n"
        )
        f.write(traceback.format_exc())
        f.write("\n")


def load_training_data():
    if not DATA.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA}")

    data = pd.read_csv(DATA, sep=r"\s+", header=None).to_numpy()
    if data.shape[1] != 37:
        raise ValueError(f"Expected 37 columns, got {data.shape[1]}")

    X = data[:, :36].astype(float)
    y = data[:, 36].astype(int)

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains non-finite feature values.")

    return X, y


def evaluate(name, cols, X_train, y_train, X_val, y_val, seed):
    cols = sorted(set(int(c) for c in cols))
    if not cols:
        raise ValueError("Feature selector returned zero features.")

    started = time.perf_counter()
    model = KNN(k=KNN_K).fit(X_train[:, cols], y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(X_val[:, cols])
    predict_seconds = time.perf_counter() - started

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": name,
        "seed": seed,
        "features": len(cols),
        "selected_indices_zero_based": ";".join(map(str, cols)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "macro_f1": float(
            f1_score(y_val, predictions, average="macro", zero_division=0)
        ),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "status": "ok",
        "error": "",
    }


def append_result(row):
    frame = pd.DataFrame([row])
    write_header = not SUMMARY.exists() or SUMMARY.stat().st_size == 0
    frame.to_csv(
        SUMMARY,
        mode="a",
        header=write_header,
        index=False,
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("Starting validation-only comparative pilot.")
    log("Official test file is not loaded or evaluated in this pilot.")

    X, y = load_training_data()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Fit preprocessing on the training portion only.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    log(
        f"Split: train={len(y_train)}, validation={len(y_val)}, "
        f"features={X_train.shape[1]}, seed={SEED}"
    )
    log(f"Classes in training split: {np.unique(y_train).tolist()}")
    log(f"Classes in validation split: {np.unique(y_val).tolist()}")

    all_cols = list(range(X_train.shape[1]))
    scores = relevance_scores(X_train, y_train)
    correlation = np.abs(np.corrcoef(X_train.T))
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    jobs = []

    jobs.append(("all_features", all_cols))

    for k in (6, 12, 18):
        top_k = np.argsort(scores)[::-1][:k].tolist()
        jobs.append((f"mutual_info_top_{k}", top_k))

    for threshold in (0.7, 0.8, 0.9, 0.95):
        graph = build_corr_graph(correlation, threshold)
        selected = select_features(scores, graph)
        jobs.append((f"heap_bfs_threshold_{threshold}", selected))

    for name, cols in jobs:
        try:
            row = evaluate(
                name, cols, X_train, y_train, X_val, y_val, SEED
            )
            append_result(row)
            log(
                f"{name}: features={row['features']}, "
                f"macro_f1={row['macro_f1']:.4f}, "
                f"accuracy={row['accuracy']:.4f}"
            )
        except Exception:
            log(f"FAILED: {name}")
            log_error(name)

    # PSO pilot: continuous positions are thresholded into a binary mask.
    # This is an initial baseline, not yet the final binary-optimizer design.
    objective_evaluations = 0
    objective_started = time.perf_counter()

    def objective(position):
        nonlocal objective_evaluations
        objective_evaluations += 1

        mask = np.asarray(position) >= 0.5
        if not mask.any():
            mask[np.argmax(position)] = True

        cols = np.flatnonzero(mask).tolist()

        try:
            row = evaluate(
                "pso_feature_selection",
                cols,
                X_train,
                y_train,
                X_val,
                y_val,
                SEED,
            )
            # Minimize validation error with a modest feature-count penalty.
            fitness = (
                0.9 * (1.0 - row["macro_f1"])
                + 0.1 * (len(cols) / X_train.shape[1])
            )
            return float(fitness)
        except Exception:
            log_error(f"PSO objective evaluation {objective_evaluations}")
            return 1.0

    try:
        problem = {
            "obj_func": objective,
            "bounds": FloatVar(
                lb=[0.0] * X_train.shape[1],
                ub=[1.0] * X_train.shape[1],
            ),
            "minmax": "min",
            "log_to": None,
        }

        optimizer = PSO.OriginalPSO(
            epoch=PSO_EPOCHS,
            pop_size=PSO_POPULATION,
        )
        optimizer.solve(problem, seed=SEED)

        best_position = np.asarray(optimizer.g_best.solution)
        best_mask = best_position >= 0.5
        if not best_mask.any():
            best_mask[np.argmax(best_position)] = True
        best_cols = np.flatnonzero(best_mask).tolist()

        row = evaluate(
            "pso_feature_selection",
            best_cols,
            X_train,
            y_train,
            X_val,
            y_val,
            SEED,
        )
        row["objective_evaluations"] = objective_evaluations
        row["optimizer_epochs"] = PSO_EPOCHS
        row["optimizer_population"] = PSO_POPULATION
        row["optimizer_best_fitness"] = float(
            optimizer.g_best.target.fitness
        )
        row["optimizer_elapsed_seconds"] = (
            time.perf_counter() - objective_started
        )

        append_result(row)
        log(
            f"PSO completed: features={row['features']}, "
            f"validation_macro_f1={row['macro_f1']:.4f}, "
            f"best_fitness={row['optimizer_best_fitness']:.6f}, "
            f"objective_evaluations={objective_evaluations}"
        )

        # Record MEALPY's global-best history when exposed by this version.
        history = getattr(optimizer, "history", None)
        best_history = getattr(history, "list_global_best_fit", None)
        if best_history is not None:
            with LOG.open("a", encoding="utf-8") as f:
                f.write("PSO global-best fitness history:\n")
                for epoch, fitness in enumerate(best_history, start=1):
                    f.write(f"epoch={epoch}, best_fitness={fitness}\n")

    except Exception:
        log("FAILED: PSO pilot")
        log_error("PSO pilot")

    log(f"Summary CSV: {SUMMARY}")
    log(f"Append-only log: {LOG}")
    log(f"Error log: {ERROR_LOG}")
    log("Pilot finished. No official test-set metrics were calculated.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_error("Fatal pilot error")
        raise
