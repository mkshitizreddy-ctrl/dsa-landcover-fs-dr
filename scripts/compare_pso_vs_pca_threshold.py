import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_pso_pca_svd import (append_log, fit_svd_projection, inner_split, knn_scores,
                             load_landsat_train, optimize_component_count)
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "results" / "pso_vs_threshold_runs.txt"


def threshold_count(singular_values, threshold):
    explained = np.square(singular_values)
    cumulative = np.cumsum(explained) / np.sum(explained)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def check_settings(folds, epochs, population, validation_fraction, variance_threshold, k_neighbors):
    if folds < 2 or epochs < 1 or population < 5:
        raise ValueError("folds >= 2, epochs >= 1, and Mealpy population >= 5 are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")


def choose_counts(X_train, y_train, X_valid, y_valid, settings, seed, log_path, run_id, label, show):
    _, _, singular_values = fit_svd_projection(X_train, X_valid)
    pso = optimize_component_count(X_train, y_train, X_valid, y_valid, seed=seed, epochs=settings["epochs"],
                                   population=settings["population"], k_neighbors=settings["k_neighbors"],
                                   n_jobs=settings["n_jobs"], log_path=log_path, run_id=run_id, label=label,
                                   show=show)
    return threshold_count(singular_values, settings["variance_threshold"]), pso["best_k"]


def compare_fold(dataset, train_idx, valid_idx, fold, settings, log_path, run_id, show):
    fold_started = time.perf_counter()
    inner = inner_split(dataset, train_idx, settings["validation_fraction"], settings["seed"] + fold)
    threshold_k, pso_k = choose_counts(inner.X_train, inner.y_train, inner.X_valid, inner.y_valid, settings,
                                       settings["seed"] + fold, log_path, run_id, f"fold_{fold}_pso_search", show)
    train_scores, valid_scores, _ = fit_svd_projection(dataset.X[train_idx], dataset.X[valid_idx])
    record = {"fold": fold}
    for method, k in (("threshold", threshold_k), ("pso", pso_k)):
        result = knn_scores(train_scores, dataset.y[train_idx], valid_scores, dataset.y[valid_idx],
                            k, settings["k_neighbors"], settings["n_jobs"])
        record.update({f"{method}_components": k, **{f"{method}_{name}": v for name, v in result.items()}})
    append_log(log_path,
               f"FOLD_RESULT run={run_id} fold={fold}/{settings['folds']} threshold_k={threshold_k} "
               f"threshold_macro_f1={record['threshold_macro_f1']:.6f} pso_k={pso_k} "
               f"pso_macro_f1={record['pso_macro_f1']:.6f} "
               f"paired_macro_f1_delta={record['pso_macro_f1'] - record['threshold_macro_f1']:+.6f} "
               f"threshold_accuracy={record['threshold_accuracy']:.6f} pso_accuracy={record['pso_accuracy']:.6f} "
               f"elapsed_seconds={time.perf_counter() - fold_started:.2f}", show)
    return record


def compare_pso_vs_threshold(folds=3, seed=42, epochs=5, population=8, validation_fraction=0.2,
                             variance_threshold=0.95, k_neighbors=5, n_jobs=-1, log_path=LOG_PATH, show=print):
    check_settings(folds, epochs, population, validation_fraction, variance_threshold, k_neighbors)
    settings = {"folds": folds, "seed": seed, "epochs": epochs, "population": population, "n_jobs": n_jobs,
                "validation_fraction": validation_fraction, "variance_threshold": variance_threshold,
                "k_neighbors": k_neighbors}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started = time.perf_counter()
    dataset = load_landsat_train()
    append_log(log_path,
               f"START run={run_id} rows={len(dataset.y)} features={dataset.X.shape[1]} folds={folds} seed={seed} "
               f"variance_threshold={variance_threshold} PSO_epochs={epochs} PSO_population={population} "
               f"classifier=sklearn.KNeighborsClassifier(k={k_neighbors}) same_outer_folds=same inner "
               "training data=SVD fit on training only official_sat.tst=NOT_ACCESSED", show)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    records = pd.DataFrame([compare_fold(dataset, train_idx, valid_idx, fold, settings, log_path, run_id, show)
                            for fold, (train_idx, valid_idx) in enumerate(splitter.split(dataset.X, dataset.y),
                                                                          start=1)])
    full = make_validation_split(dataset, validation_fraction=validation_fraction, seed=seed, scale=False)
    final_threshold_k, final_pso_k = choose_counts(full.X_train, full.y_train, full.X_valid, full.y_valid,
                                                   settings, seed, log_path, run_id, "full_training_pso_search", show)
    deltas = records["pso_macro_f1"] - records["threshold_macro_f1"]
    append_log(log_path,
               f"SUMMARY run={run_id} threshold_k_by_fold={records['threshold_components'].tolist()} "
               f"pso_k_by_fold={records['pso_components'].tolist()} "
               f"threshold_macro_f1_mean={records['threshold_macro_f1'].mean():.6f} "
               f"threshold_macro_f1_sample_sd={records['threshold_macro_f1'].std(ddof=1):.6f} "
               f"pso_macro_f1_mean={records['pso_macro_f1'].mean():.6f} "
               f"pso_macro_f1_sample_sd={records['pso_macro_f1'].std(ddof=1):.6f} "
               f"paired_delta_mean={deltas.mean():+.6f} PSO_wins={int((deltas > 0).sum())}/{folds} "
               f"full_training_threshold_k={final_threshold_k} full_training_PSO_k={final_pso_k} "
               "NOTE=outer-fold_metrics_are_comparable; final_inner_scores_are_for_selection_only", show)
    append_log(log_path, f"END run={run_id} total_elapsed_seconds={time.perf_counter() - started:.2f} "
                         "official_sat.tst=NOT_ACCESSED", show)
    return records, {"final_threshold_k": final_threshold_k, "final_pso_k": final_pso_k}
