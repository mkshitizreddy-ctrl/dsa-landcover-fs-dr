import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mealpy import PSO, FloatVar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dataset_loader import DatasetConfig, load_dataset
from validation_pipeline import make_validation_split

ROOT = Path(__file__).resolve().parents[1]
LANDSAT_TRAIN = ROOT / "data" / "sat.trn"
LOG_PATH = ROOT / "results" / "pso_pca_svd_runs.txt"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(path, message, show=print):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    show(line)


def load_landsat_train(path=LANDSAT_TRAIN):
    return load_dataset(DatasetConfig(path=str(path), delimiter=r"\s+", header=None,
                                      target_column=-1, task="classification"))


def inner_split(dataset, train_idx, fraction, seed):
    inner_dataset = type(dataset)(X=dataset.X[train_idx], y=dataset.y[train_idx],
                                  feature_names=dataset.feature_names, target_name=dataset.target_name,
                                  class_labels=dataset.class_labels, source_path=dataset.source_path)
    return make_validation_split(inner_dataset, validation_fraction=fraction, seed=seed, scale=False)


def fit_svd_projection(X_train, X_valid):
    scaler = StandardScaler().fit(X_train)
    train_scaled, valid_scaled = scaler.transform(X_train), scaler.transform(X_valid)
    _, singular_values, right_vectors = np.linalg.svd(train_scaled, full_matrices=False)
    components = right_vectors[:min(train_scaled.shape)]
    return train_scaled @ components.T, valid_scaled @ components.T, singular_values


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def knn_scores(train_scores, y_train, valid_scores, y_valid, k, k_neighbors, n_jobs):
    classifier = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=n_jobs)
    classifier.fit(train_scores[:, :k], y_train)
    return metrics(y_valid, classifier.predict(valid_scores[:, :k]))


class ComponentCountSearch:
    def __init__(self, train_scores, y_train, valid_scores, y_valid, k_neighbors, n_jobs, log):
        self.train_scores, self.y_train = train_scores, y_train
        self.valid_scores, self.y_valid = valid_scores, y_valid
        self.k_neighbors, self.n_jobs, self.log = k_neighbors, n_jobs, log
        self.max_components = train_scores.shape[1]
        self.cache = {}
        self.calls = 0
        self.best_k = None
        self.best_macro_f1 = -1.0

    def decode(self, position):
        return int(np.clip(np.rint(float(np.asarray(position).reshape(-1)[0])), 1, self.max_components))

    def consider(self, k, value):
        tied = np.isclose(value, self.best_macro_f1, rtol=0, atol=1e-12)
        if value > self.best_macro_f1 or (tied and (self.best_k is None or k < self.best_k)):
            self.best_k, self.best_macro_f1 = k, value

    def score_count(self, k):
        if k not in self.cache:
            value = knn_scores(self.train_scores, self.y_train, self.valid_scores, self.y_valid,
                               k, self.k_neighbors, self.n_jobs)["macro_f1"]
            self.cache[k] = value
            self.consider(k, value)
        return self.cache[k]

    def __call__(self, position):
        self.calls += 1
        k = self.decode(position)
        macro_f1 = self.score_count(k)
        self.log(f"evaluation={self.calls} k={k} macro_f1={macro_f1:.6f} best_k={self.best_k} "
                 f"best_macro_f1={self.best_macro_f1:.6f} unique_k_scored={len(self.cache)}")
        return 1.0 - macro_f1


def optimize_component_count(X_train, y_train, X_valid, y_valid, *, seed, epochs, population, k_neighbors,
                             n_jobs, log_path, run_id, label, show=print):
    train_scores, valid_scores, singular_values = fit_svd_projection(X_train, X_valid)
    search = ComponentCountSearch(train_scores, y_train, valid_scores, y_valid, k_neighbors, n_jobs,
                                  lambda message: append_log(log_path, f"run={run_id} stage={label} {message}", show))
    problem = {"obj_func": search, "bounds": FloatVar(lb=[1.0], ub=[float(search.max_components)]),
               "minmax": "min", "log_to": None}
    optimizer = PSO.OriginalPSO(epoch=epochs, pop_size=population)
    optimizer.solve(problem, seed=seed)
    returned_k = search.decode(optimizer.g_best.solution)
    search.consider(returned_k, search.score_count(returned_k))
    return {"best_k": search.best_k, "best_macro_f1": search.best_macro_f1, "objective_calls": search.calls,
            "unique_k_scored": len(search.cache), "max_components": search.max_components,
            "singular_values": singular_values}


def check_settings(folds, epochs, population, k_neighbors, validation_fraction):
    if folds < 2 or epochs < 1 or population < 5:
        raise ValueError("folds >= 2, epochs >= 1, and Mealpy population >= 5 are required")
    if k_neighbors < 1 or not 0 < validation_fraction < 1:
        raise ValueError("k_neighbors must be positive and validation_fraction in (0, 1)")


def evaluate_fold(dataset, train_idx, valid_idx, fold, settings, log_path, run_id, show):
    fold_started = time.perf_counter()
    inner = inner_split(dataset, train_idx, settings["validation_fraction"], settings["seed"] + fold)
    search = optimize_component_count(inner.X_train, inner.y_train, inner.X_valid, inner.y_valid,
                                      seed=settings["seed"] + fold, epochs=settings["epochs"],
                                      population=settings["population"], k_neighbors=settings["k_neighbors"],
                                      n_jobs=settings["n_jobs"], log_path=log_path, run_id=run_id,
                                      label=f"outer_fold_{fold}_search", show=show)
    train_scores, valid_scores, _ = fit_svd_projection(dataset.X[train_idx], dataset.X[valid_idx])
    result = knn_scores(train_scores, dataset.y[train_idx], valid_scores, dataset.y[valid_idx],
                        search["best_k"], settings["k_neighbors"], settings["n_jobs"])
    append_log(log_path,
               f"OUTER_RESULT run={run_id} fold={fold}/{settings['folds']} "
               f"selected_components={search['best_k']}/{search['max_components']} "
               f"inner_macro_f1={search['best_macro_f1']:.6f} outer_accuracy={result['accuracy']:.6f} "
               f"outer_macro_f1={result['macro_f1']:.6f} "
               f"outer_balanced_accuracy={result['balanced_accuracy']:.6f} "
               f"objective_calls={search['objective_calls']} unique_k_scored={search['unique_k_scored']} "
               f"elapsed_seconds={time.perf_counter() - fold_started:.2f}", show)
    return {"fold": fold, "selected_k": search["best_k"], **result}


def run_pso_pca_svd(folds=3, seed=42, epochs=5, population=8, validation_fraction=0.2, k_neighbors=5,
                    n_jobs=-1, log_path=LOG_PATH, show=print):
    check_settings(folds, epochs, population, k_neighbors, validation_fraction)
    settings = {"folds": folds, "seed": seed, "epochs": epochs, "population": population,
                "validation_fraction": validation_fraction, "k_neighbors": k_neighbors, "n_jobs": n_jobs}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started = time.perf_counter()
    try:
        dataset = load_landsat_train()
        append_log(log_path,
                   f"START run={run_id} dataset={LANDSAT_TRAIN} rows={len(dataset.y)} "
                   f"features={dataset.X.shape[1]} classes={len(dataset.class_labels)} folds={folds} seed={seed} "
                   f"epochs={epochs} population={population} k_neighbors={k_neighbors} "
                   "PCA=SVD scaler=training_only classifier=sklearn.KNeighborsClassifier "
                   "selection_metric=macro_f1 official_sat.tst=NOT_ACCESSED", show)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_results = [evaluate_fold(dataset, train_idx, valid_idx, fold, settings, log_path, run_id, show)
                        for fold, (train_idx, valid_idx) in enumerate(splitter.split(dataset.X, dataset.y), start=1)]

        final_split = make_validation_split(dataset, validation_fraction=validation_fraction, seed=seed, scale=False)
        final_search = optimize_component_count(final_split.X_train, final_split.y_train, final_split.X_valid,
                                                final_split.y_valid, seed=seed, epochs=epochs, population=population,
                                                k_neighbors=k_neighbors, n_jobs=n_jobs, log_path=log_path,
                                                run_id=run_id, label="training_only_final_search", show=show)
        scores = [r["macro_f1"] for r in fold_results]
        summary = {"selected_k_by_fold": [r["selected_k"] for r in fold_results],
                   "outer_macro_f1_mean": float(np.mean(scores)),
                   "outer_macro_f1_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                   "training_only_selected_k": final_search["best_k"],
                   "training_only_selection_macro_f1": final_search["best_macro_f1"]}
        append_log(log_path,
                   f"SUMMARY run={run_id} selected_k_by_outer_fold={summary['selected_k_by_fold']} "
                   f"outer_macro_f1_mean={summary['outer_macro_f1_mean']:.6f} "
                   f"outer_macro_f1_sample_sd={summary['outer_macro_f1_sd']:.6f} "
                   f"training_only_selected_k={summary['training_only_selected_k']} "
                   f"training_only_selection_macro_f1={summary['training_only_selection_macro_f1']:.6f} "
                   "NOTE=selection_score_is_not_an_independent_test_estimate", show)
        append_log(log_path, f"END run={run_id} total_elapsed_seconds={time.perf_counter() - started:.2f} "
                             "official_sat.tst=NOT_ACCESSED", show)
        return fold_results, summary
    except Exception:
        append_log(log_path, f"FAILED run={run_id}\n{traceback.format_exc()}", show)
        raise
