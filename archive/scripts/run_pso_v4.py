
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mealpy import PSO, FloatVar
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_loader import DatasetConfig, load_dataset
from validation_pipeline import make_validation_split
from knn import KNN


# Reproducible, deliberately modest initial budget.
SEED = 42
VALIDATION_FRACTION = 0.20
K_NEIGHBORS = 5
PSO_EPOCHS = 10
PSO_POPULATION = 10

# Objective: balance predictive performance and subset size.
ERROR_WEIGHT = 0.90
SIZE_WEIGHT = 0.10

SUMMARY = ROOT / "results" / "pso_summary_v4.csv"
LOG = ROOT / "results" / "pso_study_v4.log"
ERROR_LOG = ROOT / "results" / "pso_errors_v4.log"
EVALUATIONS = ROOT / "results" / "pso_objective_evaluations_v4.csv"
HISTORY = ROOT / "results" / "pso_fitness_history_v4.csv"

RESULT_FIELDS = [
    "timestamp_utc",
    "run_id",
    "dataset_path",
    "seed",
    "n_samples",
    "n_features_total",
    "n_classes",
    "method",
    "n_features_selected",
    "selected_feature_indices",
    "classifier",
    "k_neighbors",
    "validation_fraction",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "objective_fitness_recomputed",
    "optimizer_best_fitness",
    "objective_evaluations",
    "failed_objective_evaluations",
    "optimizer_epochs",
    "optimizer_population",
    "runtime_seconds",
    "status",
    "error",
]

EVAL_FIELDS = [
    "timestamp_utc",
    "run_id",
    "evaluation",
    "n_features_selected",
    "selected_feature_indices",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "fitness",
    "status",
    "error",
]

HISTORY_FIELDS = [
    "timestamp_utc",
    "run_id",
    "evaluation",
    "best_fitness_so_far",
    "best_macro_f1",
    "best_feature_count",
]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    line = f"[{timestamp()}] {message}"
    print(line)
    with LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def append_csv(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_error(message):
    with ERROR_LOG.open("a", encoding="utf-8") as file:
        file.write(f"\n[{timestamp()}] {message}\n")
        file.write(traceback.format_exc())
        file.write("\n")


def calculate_fitness(macro_f1, feature_count, total_features):
    return (
        ERROR_WEIGHT * (1.0 - macro_f1)
        + SIZE_WEIGHT * (feature_count / total_features)
    )


def evaluate_mask(mask, split, run_id, evaluation_number):
    indices = np.flatnonzero(mask).tolist()
    if not indices:
        raise ValueError("Feature mask selected zero features.")

    model = KNN(k=K_NEIGHBORS)
    model.fit(split.X_train[:, indices], split.y_train)
    predictions = model.predict(split.X_valid[:, indices])

    accuracy = float(accuracy_score(split.y_valid, predictions))
    macro_f1 = float(
        f1_score(
            split.y_valid,
            predictions,
            average="macro",
            zero_division=0,
        )
    )
    balanced_accuracy = float(
        balanced_accuracy_score(split.y_valid, predictions)
    )
    fitness = calculate_fitness(
        macro_f1,
        len(indices),
        split.X_train.shape[1],
    )

    row = {
        "timestamp_utc": timestamp(),
        "run_id": run_id,
        "evaluation": evaluation_number,
        "n_features_selected": len(indices),
        "selected_feature_indices": ";".join(map(str, indices)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "fitness": float(fitness),
        "status": "success",
        "error": "",
    }
    return row


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started = time.perf_counter()

    objective_evaluations = 0
    failed_evaluations = 0
    best_fitness = float("inf")
    best_metrics = None
    best_mask = None
    fitness_history = []

    try:
        data_path = ROOT / "data" / "sat.trn"
        dataset = load_dataset(
            DatasetConfig(
                path=str(data_path),
                delimiter=r"\s+",
                header=None,
                target_column=-1,
                task="classification",
            )
        )
        split = make_validation_split(
            dataset,
            validation_fraction=VALIDATION_FRACTION,
            seed=SEED,
            scale=True,
        )

        n_features = split.X_train.shape[1]

        log(f"Run ID: {run_id}")
        log(f"Dataset: {data_path}")
        log(f"Samples: {len(dataset.y)}")
        log(f"Features: {n_features}")
        log(f"Classes: {dataset.class_labels}")
        log(f"Training rows: {len(split.y_train)}")
        log(f"Validation rows: {len(split.y_valid)}")
        log(f"PSO epochs: {PSO_EPOCHS}")
        log(f"PSO population: {PSO_POPULATION}")
        log(f"Objective: {ERROR_WEIGHT}*(1-macro_f1) + "
            f"{SIZE_WEIGHT}*(selected_features/total_features)")

        def objective(position):
            nonlocal objective_evaluations
            nonlocal failed_evaluations
            nonlocal best_fitness
            nonlocal best_metrics
            nonlocal best_mask

            objective_evaluations += 1
            mask = np.asarray(position, dtype=float) >= 0.5

            # Prevent an empty subset from being evaluated.
            if not mask.any():
                mask[int(np.argmax(position))] = True

            try:
                metrics = evaluate_mask(
                    mask,
                    split,
                    run_id,
                    objective_evaluations,
                )
                append_csv(EVALUATIONS, EVAL_FIELDS, metrics)

                fitness = metrics["fitness"]
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_metrics = metrics.copy()
                    best_mask = mask.copy()

                fitness_history.append({
                    "timestamp_utc": timestamp(),
                    "run_id": run_id,
                    "evaluation": objective_evaluations,
                    "best_fitness_so_far": best_fitness,
                    "best_macro_f1": best_metrics["macro_f1"],
                    "best_feature_count": best_metrics[
                        "n_features_selected"
                    ],
                })
                append_csv(HISTORY, HISTORY_FIELDS, fitness_history[-1])

                return float(fitness)

            except Exception as exc:
                failed_evaluations += 1
                append_csv(
                    EVALUATIONS,
                    EVAL_FIELDS,
                    {
                        "timestamp_utc": timestamp(),
                        "run_id": run_id,
                        "evaluation": objective_evaluations,
                        "n_features_selected": int(mask.sum()),
                        "selected_feature_indices": ";".join(
                            map(str, np.flatnonzero(mask).tolist())
                        ),
                        "status": "error",
                        "error": repr(exc),
                    },
                )
                with ERROR_LOG.open("a", encoding="utf-8") as file:
                    file.write(
                        f"[{timestamp()}] Objective evaluation "
                        f"{objective_evaluations} failed: {exc!r}\n"
                    )
                # Finite penalty; failure is separately recorded.
                return 1.0

        problem = {
            "obj_func": objective,
            "bounds": FloatVar(
                lb=[0.0] * n_features,
                ub=[1.0] * n_features,
            ),
            "minmax": "min",
            "log_to": None,
        }

        optimizer = PSO.OriginalPSO(
            epoch=PSO_EPOCHS,
            pop_size=PSO_POPULATION,
        )
        optimizer.solve(problem, seed=SEED)

        if best_mask is None or best_metrics is None:
            raise RuntimeError(
                "PSO did not produce a successful objective evaluation."
            )

        # Re-evaluate the best mask for the final reported validation metrics.
        final_metrics = evaluate_mask(
            best_mask,
            split,
            run_id,
            objective_evaluations + 1,
        )
        selected_indices = np.flatnonzero(best_mask).tolist()
        elapsed = time.perf_counter() - started

        result = {
            "timestamp_utc": timestamp(),
            "run_id": run_id,
            "dataset_path": str(data_path),
            "seed": SEED,
            "n_samples": len(dataset.y),
            "n_features_total": n_features,
            "n_classes": len(dataset.class_labels),
            "method": "pso_binary_mask",
            "n_features_selected": len(selected_indices),
            "selected_feature_indices": ";".join(map(str, selected_indices)),
            "classifier": "custom KNN",
            "k_neighbors": K_NEIGHBORS,
            "validation_fraction": VALIDATION_FRACTION,
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "objective_fitness_recomputed": final_metrics["fitness"],
            "optimizer_best_fitness": float(best_fitness),
            "objective_evaluations": objective_evaluations,
            "failed_objective_evaluations": failed_evaluations,
            "optimizer_epochs": PSO_EPOCHS,
            "optimizer_population": PSO_POPULATION,
            "runtime_seconds": elapsed,
            "status": "success",
            "error": "",
        }
        append_csv(SUMMARY, RESULT_FIELDS, result)

        log("PSO run completed.")
        log(f"Selected features: {len(selected_indices)} / {n_features}")
        log(f"Selected zero-based indices: {selected_indices}")
        log(f"Validation macro-F1: {final_metrics['macro_f1']:.6f}")
        log(f"Validation accuracy: {final_metrics['accuracy']:.6f}")
        log(
            f"Validation balanced accuracy: "
            f"{final_metrics['balanced_accuracy']:.6f}"
        )
        log(f"Best observed objective: {best_fitness:.6f}")
        log(f"Objective evaluations: {objective_evaluations}")
        log(f"Failed objective evaluations: {failed_evaluations}")
        log(f"Elapsed seconds: {elapsed:.3f}")
        log(f"Summary CSV: {SUMMARY}")
        log(f"Evaluation CSV: {EVALUATIONS}")
        log(f"Fitness history CSV: {HISTORY}")
        log("Official sat.tst was not loaded or evaluated.")

    except Exception:
        log("PSO V4 run failed.")
        log_error("PSO V4 main")
        raise


if __name__ == "__main__":
    main()
