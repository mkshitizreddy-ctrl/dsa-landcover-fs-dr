
"""Versioned, configurable comparative runner for classification datasets."""

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repository root is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

from src.dataset_loader import DatasetConfig, load_dataset
from src.validation_pipeline import make_validation_split


RESULTS_PATH = ROOT / "results" / "comparative_summary_v1.csv"
LOG_PATH = ROOT / "results" / "comparative_study_v1.log"
ERROR_PATH = ROOT / "results" / "algorithm_errors_v1.log"

# Keep this schema fixed throughout version 1.
COLUMNS = [
    "timestamp_utc",
    "run_id",
    "dataset_path",
    "n_samples",
    "n_features_total",
    "n_classes",
    "method",
    "n_features_selected",
    "selected_feature_indices",
    "classifier",
    "k_neighbors",
    "validation_fraction",
    "seed",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "runtime_seconds",
    "status",
    "error",
]


def append_log(path, message):
    """Append a timestamped message without overwriting prior logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"{message}\n"
        )


def append_result(row):
    """Append one result and refuse to mix incompatible CSV schemas."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = (
        RESULTS_PATH.exists()
        and RESULTS_PATH.stat().st_size > 0
    )

    if exists:
        with RESULTS_PATH.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)

        if existing_header != COLUMNS:
            raise ValueError(
                f"Existing CSV schema does not match version 1: "
                f"{RESULTS_PATH}. Existing data was not overwritten. "
                "Use a new versioned output file."
            )

    with RESULTS_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_target(value):
    """Interpret an integer string as a column position; otherwise a name."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare all features against mutual-information "
            "feature subsets on a classification dataset."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--delimiter",
        choices=["whitespace", "comma", "tab"],
        default="whitespace",
        help="Input delimiter.",
    )
    parser.add_argument(
        "--header",
        choices=["none", "infer"],
        default="none",
        help="Whether the first row contains column names.",
    )
    parser.add_argument(
        "--target",
        default="-1",
        help="Target column index or column name. Default: last column.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN.",
    )
    parser.add_argument(
        "--top-k",
        default="6,12,18",
        help="Comma-separated feature counts for MI selection.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.k_neighbors < 1:
        raise ValueError("--k-neighbors must be at least 1.")

    if not 0 < args.validation_fraction < 1:
        raise ValueError(
            "--validation-fraction must be greater than 0 and less than 1."
        )

    delimiter_map = {
        "whitespace": r"\s+",
        "comma": ",",
        "tab": "\t",
    }

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    dataset = load_dataset(
        DatasetConfig(
            path=str(data_path),
            delimiter=delimiter_map[args.delimiter],
            header=0 if args.header == "infer" else None,
            target_column=parse_target(args.target),
            task="classification",
        )
    )

    split = make_validation_split(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        scale=True,
    )

    n_features = split.X_train.shape[1]
    n_classes = len(np.unique(dataset.y))

    try:
        top_ks = sorted({
            int(part.strip())
            for part in args.top_k.split(",")
            if part.strip()
        })
    except ValueError as exc:
        raise ValueError(
            "--top-k must be comma-separated positive integers."
        ) from exc

    if any(k < 1 for k in top_ks):
        raise ValueError("Every --top-k value must be positive.")

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    append_log(
        LOG_PATH,
        f"START run_id={run_id} "
        f"data={dataset.source_path} "
        f"samples={len(dataset.y)} "
        f"features={n_features} "
        f"classes={n_classes} "
        f"seed={args.seed}",
    )

    print("Dataset:", dataset.source_path)
    print("Samples:", len(dataset.y))
    print("Features:", n_features)
    print("Classes:", dataset.class_labels)
    print("Training rows:", len(split.y_train))
    print("Validation rows:", len(split.y_valid))
    print("Run ID:", run_id)

    # Fit feature ranking using training rows only.
    mi_scores = mutual_info_classif(
        split.X_train,
        split.y_train,
        random_state=args.seed,
    )
    ranking = np.argsort(-mi_scores, kind="stable")

    experiments = [
        ("all_features", np.arange(n_features))
    ]

    for k in top_ks:
        if k >= n_features:
            append_log(
                LOG_PATH,
                f"SKIP run_id={run_id} method=mi_top_{k}: "
                f"requested {k} features, dataset has {n_features}",
            )
            print(
                f"Skipping MI top-{k}: "
                f"dataset has only {n_features} features."
            )
            continue

        experiments.append((f"mi_top_{k}", ranking[:k]))

    for method, selected in experiments:
        started = time.perf_counter()

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "dataset_path": dataset.source_path,
            "n_samples": len(dataset.y),
            "n_features_total": n_features,
            "n_classes": n_classes,
            "method": method,
            "n_features_selected": len(selected),
            "selected_feature_indices": json.dumps(
                [int(index) for index in selected.tolist()]
            ),
            "classifier": "sklearn.KNeighborsClassifier",
            "k_neighbors": args.k_neighbors,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "accuracy": "",
            "macro_f1": "",
            "balanced_accuracy": "",
            "runtime_seconds": "",
            "status": "error",
            "error": "",
        }

        try:
            model = KNeighborsClassifier(
                n_neighbors=args.k_neighbors
            )
            model.fit(
                split.X_train[:, selected],
                split.y_train,
            )
            predictions = model.predict(
                split.X_valid[:, selected]
            )

            row.update({
                "accuracy": accuracy_score(
                    split.y_valid, predictions
                ),
                "macro_f1": f1_score(
                    split.y_valid,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    split.y_valid, predictions
                ),
                "status": "success",
            })

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"

            append_log(
                ERROR_PATH,
                f"run_id={run_id} method={method} "
                f"error={row['error']}\n"
                f"{traceback.format_exc()}",
            )

        row["runtime_seconds"] = round(
            time.perf_counter() - started, 6
        )

        # Persist each experiment independently.
        try:
            append_result(row)
        except Exception as exc:
            append_log(
                ERROR_PATH,
                f"Could not write result for run_id={run_id} "
                f"method={method}: {type(exc).__name__}: {exc}",
            )
            raise

        append_log(
            LOG_PATH,
            f"RESULT run_id={run_id} "
            f"method={method} "
            f"status={row['status']} "
            f"features={row['n_features_selected']} "
            f"accuracy={row['accuracy']} "
            f"macro_f1={row['macro_f1']} "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"seconds={row['runtime_seconds']}",
        )

        print(
            f"{method}: status={row['status']}, "
            f"features={row['n_features_selected']}, "
            f"macro-F1={row['macro_f1']}, "
            f"runtime={row['runtime_seconds']}s"
        )

    append_log(LOG_PATH, f"END run_id={run_id}")

    print("\nResults appended to:", RESULTS_PATH)
    print("Run log appended to:", LOG_PATH)
    print("Error log:", ERROR_PATH)


if __name__ == "__main__":
    main()
