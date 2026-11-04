import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from dataset_loader import DatasetConfig, load_dataset
from graph import build_corr_graph
from mi import relevance_scores
from selection import select_features
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
LANDSAT_TRAIN = ROOT / "data" / "sat.trn"
RESULTS_PATH = ROOT / "results" / "comparative_summary_v3.csv"
LOG_PATH = ROOT / "results" / "comparative_study_v3.log"
ERROR_PATH = ROOT / "results" / "algorithm_errors_v3.log"
RANKING_PREFIXES = (("mi", "mi_top"), ("anova", "anova_top"), ("variance", "variance_top"),
                    ("random_forest", "random_forest_top"), ("rfe", "rfe_top"))

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
            if next(csv.reader(f), None) != COLUMNS:
                raise ValueError("Existing v3 CSV has an unexpected schema; refusing to overwrite it.")
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def check_ranking(name, ranking, n_features):
    if len(ranking) != n_features or not np.array_equal(np.sort(ranking), np.arange(n_features)):
        raise ValueError(f"{name} produced an invalid feature ranking.")


def rank_classical_features(X_train, y_train, X_raw_train, seed):
    rankings = {}
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    rankings["mi"] = np.argsort(-mi, kind="stable")

    anova, _ = f_classif(X_train, y_train)
    anova = np.nan_to_num(anova, nan=-np.inf, posinf=1e300, neginf=-np.inf)
    rankings["anova"] = np.argsort(-anova, kind="stable")

    rankings["variance"] = np.argsort(-np.var(X_raw_train, axis=0), kind="stable")

    forest = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    rankings["random_forest"] = np.argsort(-forest.fit(X_train, y_train).feature_importances_, kind="stable")

    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1,
                                               class_weight="balanced"),
              n_features_to_select=1, step=1)
    rankings["rfe"] = np.argsort(rfe.fit(X_train, y_train).ranking_, kind="stable")

    for name, ranking in rankings.items():
        check_ranking(name, ranking, X_train.shape[1])
    return rankings


def check_subset(selected, n_features, label):
    if selected.size == 0:
        raise ValueError(f"Empty feature set ({label}).")
    if len(np.unique(selected)) != len(selected):
        raise ValueError(f"Duplicate feature indices ({label}).")
    if np.any(selected < 0) or np.any(selected >= n_features):
        raise ValueError(f"Feature index out of range ({label}).")


def graph_experiments(X_train, y_train, thresholds):
    correlations = np.nan_to_num(np.corrcoef(X_train, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlations, 1.0)
    scores = relevance_scores(X_train, y_train)
    experiments = []
    for t in thresholds:
        selected = np.asarray(sorted(select_features(scores, build_corr_graph(correlations, t))), dtype=int)
        check_subset(selected, X_train.shape[1], f"graph selector at {t}")
        experiments.append((f"graph_heap_bfs_t{t:g}", selected))
    return experiments


def build_experiments(split, raw_X_train, y_train, seed, top_ks, thresholds):
    X_train = np.asarray(split.X_train, dtype=float)
    n_features = X_train.shape[1]
    rankings = rank_classical_features(X_train, y_train, raw_X_train, seed)
    experiments = [("all_features", np.arange(n_features))]
    for k in top_ks:
        if k < n_features:
            experiments += [(f"{prefix}_{k}", rankings[key][:k]) for key, prefix in RANKING_PREFIXES]
    return experiments + graph_experiments(X_train, y_train, thresholds)


def score_subset(split, selected, k_neighbors):
    check_subset(selected, split.X_train.shape[1], "evaluation")
    if k_neighbors > len(split.y_train):
        raise ValueError("k-neighbors exceeds training sample count.")
    model = KNeighborsClassifier(n_neighbors=k_neighbors).fit(split.X_train[:, selected], split.y_train)
    pred = model.predict(split.X_valid[:, selected])
    return {"accuracy": float(accuracy_score(split.y_valid, pred)),
            "macro_f1": float(f1_score(split.y_valid, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(split.y_valid, pred))}


def evaluate_one(dataset, split, run_id, method, selected, settings, save):
    started = time.perf_counter()
    selected = np.asarray(selected, dtype=int)
    row = {c: "" for c in COLUMNS}
    row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id,
               dataset_path=dataset.source_path, n_samples=len(dataset.y),
               n_features_total=split.X_train.shape[1], n_classes=len(np.unique(dataset.y)), method=method,
               n_features_selected=len(selected), selected_feature_indices=json.dumps(selected.tolist()),
               classifier="sklearn.KNeighborsClassifier", k_neighbors=settings["k_neighbors"],
               validation_fraction=settings["validation_fraction"], seed=settings["seed"], status="error")
    try:
        row.update(score_subset(split, selected, settings["k_neighbors"]), status="success")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if save:
            append_log(ERROR_PATH, f"run_id={run_id} method={method} error={row['error']}\n{traceback.format_exc()}")
    row["runtime_seconds"] = round(time.perf_counter() - started, 6)
    if save:
        append_result(row)
        append_log(LOG_PATH, f"RESULT run_id={run_id} method={method} status={row['status']} "
                             f"features={row['n_features_selected']} macro_f1={row['macro_f1']} "
                             f"seconds={row['runtime_seconds']}")
    return row


def check_settings(k_neighbors, validation_fraction, top_ks, thresholds):
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be at least 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError("top_ks must contain positive integers.")
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("thresholds must be between 0 and 1.")


def run_comparative_v3(data_path=LANDSAT_TRAIN, delimiter=r"\s+", header=None, target=-1,
                       validation_fraction=0.2, seed=42, k_neighbors=5, top_ks=(6, 12, 18),
                       thresholds=(0.7, 0.8, 0.9, 0.95), save=True, show=print):
    top_ks, thresholds = sorted(set(top_ks)), sorted(set(thresholds))
    check_settings(k_neighbors, validation_fraction, top_ks, thresholds)
    dataset = load_dataset(DatasetConfig(path=str(data_path), delimiter=delimiter, header=header,
                                         target_column=target, task="classification"))
    split = make_validation_split(dataset, validation_fraction=validation_fraction, seed=seed, scale=True)
    raw_X_train = np.asarray(dataset.X, dtype=float)[np.asarray(split.train_indices, dtype=int)]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    settings = {"k_neighbors": k_neighbors, "validation_fraction": validation_fraction, "seed": seed}
    if save:
        append_log(LOG_PATH, f"START run_id={run_id} data={dataset.source_path} seed={seed} "
                             f"validation_fraction={validation_fraction}")
    show(f"Samples {len(dataset.y)}, features {split.X_train.shape[1]}, training rows {len(split.y_train)}, "
         f"validation rows {len(split.y_valid)}")

    experiments = build_experiments(split, raw_X_train, np.asarray(split.y_train), seed, top_ks, thresholds)
    rows = []
    for method, selected in experiments:
        row = evaluate_one(dataset, split, run_id, method, selected, settings, save)
        rows.append(row)
        show(f"{method}: {row['status']}, features={row['n_features_selected']}, macro-F1={row['macro_f1']}")

    failures = sum(r["status"] != "success" for r in rows)
    if save:
        append_log(LOG_PATH, f"END run_id={run_id} experiments={len(rows)} successes={len(rows) - failures} "
                             f"failures={failures}")
    return pd.DataFrame(rows, columns=COLUMNS)
