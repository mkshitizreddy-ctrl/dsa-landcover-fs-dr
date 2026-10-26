
"""V3 comparative runner: classical ranking + graph/heap/BFS selection.

Uses a stratified training/validation split. The official test set is not read.
Results are appended to versioned v3 files; v1/v2 and original scripts remain intact.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier

from src.dataset_loader import DatasetConfig, load_dataset
from src.validation_pipeline import make_validation_split
from graph import build_corr_graph
from mi import relevance_scores
from selection import select_features

RESULTS_PATH = ROOT / "results" / "comparative_summary_v3.csv"
LOG_PATH = ROOT / "results" / "comparative_study_v3.log"
ERROR_PATH = ROOT / "results" / "algorithm_errors_v3.log"

COLUMNS = [
    "timestamp_utc", "run_id", "dataset_path", "n_samples",
    "n_features_total", "n_classes", "method", "n_features_selected",
    "selected_feature_indices", "classifier", "k_neighbors",
    "validation_fraction", "seed", "accuracy", "macro_f1",
    "balanced_accuracy", "runtime_seconds", "status", "error",
]


def append_log(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")


def append_result(row):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0

    if exists:
        with RESULTS_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), None)
        if header != COLUMNS:
            raise ValueError(
                "Existing v3 CSV has an unexpected schema; refusing to overwrite it."
            )

    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_target(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare classical feature ranking and graph/heap/BFS selection."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--delimiter", choices=["whitespace", "comma", "tab"],
        default="whitespace",
    )
    parser.add_argument("--header", choices=["none", "infer"], default="none")
    parser.add_argument("--target", default="-1")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--top-k", default="6,12,18")
    parser.add_argument("--thresholds", default="0.7,0.8,0.9,0.95")
    return parser.parse_args()


def rank_classical_features(X_train, y_train, X_raw_train, seed):
    """Fit classical feature rankings using training rows only."""
    n_features = X_train.shape[1]
    rankings = {}

    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    rankings["mi"] = np.argsort(-mi, kind="stable")

    anova, _ = f_classif(X_train, y_train)
    # NaN scores are placed last; infinite F scores are ranked first.
    anova = np.nan_to_num(anova, nan=-np.inf, posinf=1e300, neginf=-np.inf)
    rankings["anova"] = np.argsort(-anova, kind="stable")

    variance = np.var(X_raw_train, axis=0)
    rankings["variance"] = np.argsort(-variance, kind="stable")

    forest = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    forest.fit(X_train, y_train)
    rankings["random_forest"] = np.argsort(
        -forest.feature_importances_, kind="stable"
    )

    rfe = RFE(
        estimator=RandomForestClassifier(
            n_estimators=50,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
        n_features_to_select=1,
        step=1,
    )
    rfe.fit(X_train, y_train)
    rankings["rfe"] = np.argsort(rfe.ranking_, kind="stable")

    expected = np.arange(n_features)
    for name, ranking in rankings.items():
        if len(ranking) != n_features or not np.array_equal(
            np.sort(ranking), expected
        ):
            raise ValueError(f"{name} produced an invalid feature ranking.")

    return rankings


def build_experiments(split, raw_X_train, y_train, seed, top_ks, thresholds):
    """Build every feature subset from training data only."""
    X_train = np.asarray(split.X_train, dtype=float)
    n_features = X_train.shape[1]
    rankings = rank_classical_features(
        X_train, y_train, raw_X_train, seed
    )

    experiments = [("all_features", np.arange(n_features))]

    prefixes = [
        ("mi", "mi_top"),
        ("anova", "anova_top"),
        ("variance", "variance_top"),
        ("random_forest", "random_forest_top"),
        ("rfe", "rfe_top"),
    ]

    for k in top_ks:
        if k >= n_features:
            print(f"Skipping top-{k}: must be smaller than {n_features}.")
            continue
        for ranking_key, prefix in prefixes:
            experiments.append((f"{prefix}_{k}", rankings[ranking_key][:k]))

    # Use original graph implementation. Correlation is calculated from
    # training features only; no validation rows enter graph construction.
    correlations = np.corrcoef(X_train, rowvar=False)
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlations, 1.0)

    # The existing graph builder applies abs(C[i,j]) > threshold.
    graph_scores = relevance_scores(X_train, y_train)

    for threshold in thresholds:
        graph = build_corr_graph(correlations, threshold)
        selected = np.asarray(
            sorted(select_features(graph_scores, graph)), dtype=int
        )
        if selected.size == 0:
            raise ValueError(f"Graph selector returned no features at {threshold}.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError(f"Graph selector returned duplicates at {threshold}.")
        if np.any(selected < 0) or np.any(selected >= n_features):
            raise ValueError(f"Graph selector returned invalid indices at {threshold}.")
        experiments.append(
            (f"graph_heap_bfs_t{threshold:g}", selected)
        )

    return experiments


def evaluate_one(dataset, split, run_id, method, selected, args):
    started = time.perf_counter()
    n_features = split.X_train.shape[1]
    n_classes = len(np.unique(dataset.y))
    selected = np.asarray(selected, dtype=int)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset_path": dataset.source_path,
        "n_samples": len(dataset.y),
        "n_features_total": n_features,
        "n_classes": n_classes,
        "method": method,
        "n_features_selected": len(selected),
        "selected_feature_indices": json.dumps(selected.tolist()),
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
        if selected.size == 0:
            raise ValueError("Selected feature set is empty.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected feature indices contain duplicates.")
        if np.any(selected < 0) or np.any(selected >= n_features):
            raise ValueError("Selected feature index is out of range.")
        if args.k_neighbors > len(split.y_train):
            raise ValueError("k-neighbors exceeds training sample count.")

        model = KNeighborsClassifier(n_neighbors=args.k_neighbors)
        model.fit(split.X_train[:, selected], split.y_train)
        prediction = model.predict(split.X_valid[:, selected])

        row.update({
            "accuracy": float(accuracy_score(split.y_valid, prediction)),
            "macro_f1": float(f1_score(
                split.y_valid, prediction, average="macro", zero_division=0
            )),
            "balanced_accuracy": float(
                balanced_accuracy_score(split.y_valid, prediction)
            ),
            "status": "success",
            "error": "",
        })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        append_log(
            ERROR_PATH,
            f"run_id={run_id} method={method} error={row['error']}\n"
            f"{traceback.format_exc()}",
        )

    row["runtime_seconds"] = round(time.perf_counter() - started, 6)
    append_result(row)
    append_log(
        LOG_PATH,
        f"RESULT run_id={run_id} method={method} status={row['status']} "
        f"features={row['n_features_selected']} macro_f1={row['macro_f1']} "
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
        raise ValueError("--validation-fraction must be between 0 and 1.")

    try:
        top_ks = sorted({
            int(p.strip()) for p in args.top_k.split(",") if p.strip()
        })
        thresholds = sorted({
            float(p.strip()) for p in args.thresholds.split(",") if p.strip()
        })
    except ValueError as exc:
        raise ValueError("Invalid --top-k or --thresholds value.") from exc

    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError("--top-k must contain positive integers.")
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("--thresholds must be between 0 and 1.")

    delimiter_map = {
        "whitespace": r"\s+",
        "comma": ",",
        "tab": "\t",
    }
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    dataset = load_dataset(DatasetConfig(
        path=str(data_path),
        delimiter=delimiter_map[args.delimiter],
        header=0 if args.header == "infer" else None,
        target_column=parse_target(args.target),
        task="classification",
    ))

    split = make_validation_split(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        scale=True,
    )

    # Map split row indices back to the raw dataset for variance ranking.
    raw_X = np.asarray(dataset.X, dtype=float)
    raw_X_train = raw_X[np.asarray(split.train_indices, dtype=int)]
    y_train = np.asarray(split.y_train)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    append_log(
        LOG_PATH,
        f"START run_id={run_id} data={dataset.source_path} "
        f"seed={args.seed} validation_fraction={args.validation_fraction}",
    )

    print("Dataset:", dataset.source_path)
    print("Samples:", len(dataset.y))
    print("Features:", split.X_train.shape[1])
    print("Classes:", dataset.class_labels)
    print("Training rows:", len(split.y_train))
    print("Validation rows:", len(split.y_valid))
    print("Run ID:", run_id)

    # If setup fails, log it and stop before scheduling partial experiments.
    try:
        experiments = build_experiments(
            split=split,
            raw_X_train=raw_X_train,
            y_train=y_train,
            seed=args.seed,
            top_ks=top_ks,
            thresholds=thresholds,
        )
    except Exception as exc:
        append_log(
            ERROR_PATH,
            f"run_id={run_id} setup_error={type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
        )
        raise

    print("Experiments scheduled:", len(experiments))
    successes = failures = 0

    for method, selected in experiments:
        row = evaluate_one(dataset, split, run_id, method, selected, args)
        if row["status"] == "success":
            successes += 1
        else:
            failures += 1

    append_log(
        LOG_PATH,
        f"END run_id={run_id} experiments={len(experiments)} "
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
