
"""Versioned comparative runner with classical feature-selection methods.

Methods:
- All features
- Mutual information (MI)
- ANOVA F-test
- Training-set variance ranking
- Random Forest feature importance
- Recursive Feature Elimination (RFE)

Feature ranking is fitted on training data only. Results are appended to
versioned v2 output files, leaving the v1 outputs untouched.
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

from src.dataset_loader import DatasetConfig, load_dataset
from src.validation_pipeline import make_validation_split


RESULTS_PATH = ROOT / "results" / "comparative_summary_v2.csv"
LOG_PATH = ROOT / "results" / "comparative_study_v2.log"
ERROR_PATH = ROOT / "results" / "algorithm_errors_v2.log"

# Keep the same 19-column schema used by v1.
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
                f"Existing CSV schema does not match v2: "
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
            "Compare classical feature-selection methods on a "
            "classification dataset using a training/validation split."
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
        help="Comma-separated feature counts for ranking methods.",
    )

    return parser.parse_args()


def build_feature_rankings(dataset, split, seed):
    """Build feature rankings using training rows only.

    Returns:
        A dictionary mapping method prefixes to ranked feature indices.

    Notes:
        MI, ANOVA, Random Forest, and RFE use the training features
        from the validation pipeline.

        Variance ranking uses the original unscaled features, restricted
        to training rows. It must not use validation rows.
    """
    X_train = np.asarray(split.X_train, dtype=float)
    y_train = np.asarray(split.y_train)

    n_features = X_train.shape[1]
    rankings = {}

    # Mutual information ranking.
    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=seed,
    )
    rankings["mi"] = np.argsort(-mi_scores, kind="stable")

    # ANOVA F-test ranking.
    anova_scores, _ = f_classif(X_train, y_train)
    anova_scores = np.nan_to_num(
        anova_scores,
        nan=-np.inf,
        posinf=np.finfo(float).max,
        neginf=-np.inf,
    )
    rankings["anova"] = np.argsort(
        -anova_scores,
        kind="stable",
    )

    # Variance ranking on raw, unscaled training rows only.
    raw_X = np.asarray(dataset.X, dtype=float)
    train_indices = np.asarray(split.train_indices, dtype=int)
    raw_X_train = raw_X[train_indices]

    if raw_X_train.shape[1] != n_features:
        raise ValueError(
            "Raw and processed training feature counts do not match."
        )

    variance_scores = np.var(raw_X_train, axis=0)
    rankings["variance"] = np.argsort(
        -variance_scores,
        kind="stable",
    )

    # Random Forest feature importance ranking.
    forest = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    forest.fit(X_train, y_train)

    rankings["random_forest"] = np.argsort(
        -forest.feature_importances_,
        kind="stable",
    )

    # Recursive Feature Elimination ranking.
    # RFE ranking_ is 1 for the best-ranked feature; lower is better.
    rfe = RFE(
        estimator=RandomForestClassifier(
            n_estimators=50,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        n_features_to_select=1,
        step=1,
    )
    rfe.fit(X_train, y_train)

    rankings["rfe"] = np.argsort(
        rfe.ranking_,
        kind="stable",
    )

    # Defensive validation of all ranking outputs.
    expected_indices = np.arange(n_features)

    for method, ranking in rankings.items():
        if (
            len(ranking) != n_features
            or not np.array_equal(
                np.sort(ranking), expected_indices
            )
        ):
            raise ValueError(
                f"{method} did not produce a valid feature ranking."
            )

    return rankings


def make_result_row(
    dataset,
    run_id,
    method,
    selected,
    n_features,
    n_classes,
    args,
):
    """Create a result row using the fixed v2 schema."""
    return {
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


def evaluate_experiment(
    dataset,
    split,
    run_id,
    method,
    selected,
    n_features,
    n_classes,
    args,
):
    """Evaluate and persist one experiment, recording failures."""
    started = time.perf_counter()

    row = make_result_row(
        dataset=dataset,
        run_id=run_id,
        method=method,
        selected=selected,
        n_features=n_features,
        n_classes=n_classes,
        args=args,
    )

    try:
        if len(selected) == 0:
            raise ValueError("The selected feature set is empty.")

        if args.k_neighbors > len(split.y_train):
            raise ValueError(
                f"k_neighbors={args.k_neighbors} exceeds the "
                f"training sample count ({len(split.y_train)})."
            )

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
            "accuracy": float(
                accuracy_score(split.y_valid, predictions)
            ),
            "macro_f1": float(
                f1_score(
                    split.y_valid,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    split.y_valid,
                    predictions,
                )
            ),
            "status": "success",
            "error": "",
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
        time.perf_counter() - started,
        6,
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

    return row


def main():
    args = parse_args()

    if args.k_neighbors < 1:
        raise ValueError("--k-neighbors must be at least 1.")

    if not 0 < args.validation_fraction < 1:
        raise ValueError(
            "--validation-fraction must be greater than 0 "
            "and less than 1."
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

    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError(
            "--top-k must contain positive integers, "
            "for example: --top-k 6,12,18"
        )

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

    # Compute rankings using the training split only.
    try:
        rankings = build_feature_rankings(
            dataset=dataset,
            split=split,
            seed=args.seed,
        )
    except Exception as exc:
        append_log(
            ERROR_PATH,
            f"run_id={run_id} ranking_setup_error="
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
        )
        raise

    experiments = [
        ("all_features", np.arange(n_features)),
    ]

    # Add all ranking-based methods at each requested feature count.
    ranking_methods = [
        ("mi", "mi_top"),
        ("anova", "anova_top"),
        ("variance", "variance_top"),
        ("random_forest", "random_forest_top"),
        ("rfe", "rfe_top"),
    ]

    for k in top_ks:
        if k >= n_features:
            message = (
                f"Skipping top-{k}: dataset has "
                f"{n_features} features; requested count "
                "must be smaller than the total."
            )
            append_log(LOG_PATH, f"SKIP run_id={run_id} {message}")
            print(message)
            continue

        for ranking_key, method_prefix in ranking_methods:
            selected = rankings[ranking_key][:k]
            experiments.append(
                (f"{method_prefix}_{k}", selected)
            )

    print("Experiments scheduled:", len(experiments))

    successes = 0
    failures = 0

    for method, selected in experiments:
        row = evaluate_experiment(
            dataset=dataset,
            split=split,
            run_id=run_id,
            method=method,
            selected=np.asarray(selected, dtype=int),
            n_features=n_features,
            n_classes=n_classes,
            args=args,
        )

        if row["status"] == "success":
            successes += 1
        else:
            failures += 1

    append_log(
        LOG_PATH,
        f"END run_id={run_id} "
        f"experiments={len(experiments)} "
        f"successes={successes} failures={failures}",
    )

    print("\nRun completed.")
    print("Experiments:", len(experiments))
    print("Successful:", successes)
    print("Failed:", failures)
    print("Results appended to:", RESULTS_PATH)
    print("Run log appended to:", LOG_PATH)
    print("Error log:", ERROR_PATH)


if __name__ == "__main__":
    main()
