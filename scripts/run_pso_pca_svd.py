"""Use Mealpy PSO to choose a fixed number of SVD-based PCA components.

The study uses nested stratified cross-validation on data/sat.trn. The outer
fold estimates performance; PSO searches only an inner split of that fold's
training rows. A final PSO search on a training-only split suggests one
component count for fitting a later model. The official sat.tst file is never
loaded. All run output is appended to results/pso_pca_svd_runs.txt.
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mealpy import FloatVar, PSO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_loader import DatasetConfig, load_dataset
from validation_pipeline import make_validation_split


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def fit_svd_projection(X_train, X_valid):
    """Fit scaling and PCA directions on training rows, using SVD explicitly."""
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    valid_scaled = scaler.transform(X_valid)

    # full_matrices=False gives at most min(samples, features) components.
    _, singular_values, right_vectors = np.linalg.svd(
        train_scaled, full_matrices=False
    )
    max_components = min(train_scaled.shape)
    components = right_vectors[:max_components]
    train_scores = train_scaled @ components.T
    valid_scores = valid_scaled @ components.T
    return train_scores, valid_scores, singular_values


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def optimize_component_count(
    X_train,
    y_train,
    X_valid,
    y_valid,
    *,
    seed,
    epochs,
    population,
    k_neighbors,
    n_jobs,
    log_path,
    run_id,
    label,
):
    train_scores, valid_scores, singular_values = fit_svd_projection(
        X_train, X_valid
    )
    max_components = train_scores.shape[1]
    cache = {}
    objective_calls = 0
    best_k = None
    best_macro_f1 = -1.0

    def score_count(k):
        nonlocal best_k, best_macro_f1
        if k not in cache:
            classifier = KNeighborsClassifier(
                n_neighbors=k_neighbors, n_jobs=n_jobs
            )
            classifier.fit(train_scores[:, :k], y_train)
            prediction = classifier.predict(valid_scores[:, :k])
            value = float(
                f1_score(y_valid, prediction, average="macro", zero_division=0)
            )
            cache[k] = value
            if value > best_macro_f1 or (
                np.isclose(value, best_macro_f1, rtol=0, atol=1e-12)
                and (best_k is None or k < best_k)
            ):
                best_k, best_macro_f1 = k, value
        return cache[k]

    def objective(position):
        nonlocal objective_calls
        objective_calls += 1
        k = int(np.clip(np.rint(float(np.asarray(position).reshape(-1)[0])), 1, max_components))
        macro_f1 = score_count(k)
        append_log(
            log_path,
            f"run={run_id} stage={label} evaluation={objective_calls} "
            f"k={k} macro_f1={macro_f1:.6f} "
            f"best_k={best_k} best_macro_f1={best_macro_f1:.6f} "
            f"unique_k_scored={len(cache)}",
        )
        return 1.0 - macro_f1

    problem = {
        "obj_func": objective,
        "bounds": FloatVar(lb=[1.0], ub=[float(max_components)]),
        "minmax": "min",
        "log_to": None,
    }
    optimizer = PSO.OriginalPSO(epoch=epochs, pop_size=population)
    optimizer.solve(problem, seed=seed)

    # PSO's returned position is scored too, including if rounding maps it to
    # an integer that did not appear in the objective calls.
    returned_k = int(
        np.clip(np.rint(float(np.asarray(optimizer.g_best.solution).reshape(-1)[0])),
                1, max_components)
    )
    returned_score = score_count(returned_k)
    if returned_score > best_macro_f1 or (
        np.isclose(returned_score, best_macro_f1, rtol=0, atol=1e-12)
        and returned_k < best_k
    ):
        best_k, best_macro_f1 = returned_k, returned_score

    return {
        "best_k": best_k,
        "best_macro_f1": best_macro_f1,
        "objective_calls": objective_calls,
        "unique_k_scored": len(cache),
        "max_components": max_components,
        "singular_values": singular_values,
    }


def make_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--log-path", type=Path,
        default=ROOT / "results" / "pso_pca_svd_runs.txt",
        help="Append run details here (default: results/pso_pca_svd_runs.txt)",
    )
    return parser.parse_args()


def main():
    args = make_args()
    if args.folds < 2 or args.epochs < 1 or args.population < 5:
        raise ValueError("folds >= 2, epochs >= 1, and Mealpy population >= 5 are required")
    if args.k_neighbors < 1 or not 0 < args.validation_fraction < 1:
        raise ValueError("k-neighbors must be positive and validation fraction in (0, 1)")

    log_path = args.log_path
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started = time.perf_counter()
    try:
        data_path = ROOT / "data" / "sat.trn"
        dataset = load_dataset(DatasetConfig(
            path=str(data_path), delimiter=r"\s+", header=None,
            target_column=-1, task="classification",
        ))
        append_log(
            log_path,
            f"START run={run_id} dataset={data_path} rows={len(dataset.y)} "
            f"features={dataset.X.shape[1]} classes={len(dataset.class_labels)} "
            f"folds={args.folds} seed={args.seed} epochs={args.epochs} "
            f"population={args.population} k_neighbors={args.k_neighbors} "
            "PCA=SVD scaler=training_only classifier=sklearn.KNeighborsClassifier "
            "selection_metric=macro_f1 official_sat.tst=NOT_ACCESSED",
        )
        splitter = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        )
        outer_scores = []
        selected_counts = []

        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(dataset.X, dataset.y), start=1
        ):
            fold_started = time.perf_counter()
            outer_X_train, outer_y_train = dataset.X[train_idx], dataset.y[train_idx]
            outer_X_valid, outer_y_valid = dataset.X[valid_idx], dataset.y[valid_idx]
            inner_dataset = type(dataset)(
                X=outer_X_train, y=outer_y_train,
                feature_names=dataset.feature_names,
                target_name=dataset.target_name,
                class_labels=dataset.class_labels,
                source_path=dataset.source_path,
            )
            inner = make_validation_split(
                inner_dataset, validation_fraction=args.validation_fraction,
                seed=args.seed + fold, scale=False,
            )
            search = optimize_component_count(
                inner.X_train, inner.y_train, inner.X_valid, inner.y_valid,
                seed=args.seed + fold, epochs=args.epochs,
                population=args.population, k_neighbors=args.k_neighbors,
                n_jobs=args.n_jobs, log_path=log_path, run_id=run_id,
                label=f"outer_fold_{fold}_search",
            )

            # Refit scaling and SVD on the full outer-training fold, then
            # evaluate the selected k once on the untouched outer fold.
            train_scores, valid_scores, _ = fit_svd_projection(
                outer_X_train, outer_X_valid
            )
            classifier = KNeighborsClassifier(
                n_neighbors=args.k_neighbors, n_jobs=args.n_jobs
            )
            classifier.fit(train_scores[:, :search["best_k"]], outer_y_train)
            prediction = classifier.predict(valid_scores[:, :search["best_k"]])
            fold_metrics = metrics(outer_y_valid, prediction)
            outer_scores.append(fold_metrics["macro_f1"])
            selected_counts.append(search["best_k"])
            append_log(
                log_path,
                f"OUTER_RESULT run={run_id} fold={fold}/{args.folds} "
                f"selected_components={search['best_k']}/{search['max_components']} "
                f"inner_macro_f1={search['best_macro_f1']:.6f} "
                f"outer_accuracy={fold_metrics['accuracy']:.6f} "
                f"outer_macro_f1={fold_metrics['macro_f1']:.6f} "
                f"outer_balanced_accuracy={fold_metrics['balanced_accuracy']:.6f} "
                f"objective_calls={search['objective_calls']} "
                f"unique_k_scored={search['unique_k_scored']} "
                f"elapsed_seconds={time.perf_counter() - fold_started:.2f}",
            )

        # Select one deployable k using all training data only. Its validation
        # score is for selection and is not reported as an unbiased estimate.
        final_split = make_validation_split(
            dataset, validation_fraction=args.validation_fraction,
            seed=args.seed, scale=False,
        )
        final_search = optimize_component_count(
            final_split.X_train, final_split.y_train,
            final_split.X_valid, final_split.y_valid,
            seed=args.seed, epochs=args.epochs, population=args.population,
            k_neighbors=args.k_neighbors, n_jobs=args.n_jobs,
            log_path=log_path, run_id=run_id, label="training_only_final_search",
        )

        mean_score = float(np.mean(outer_scores))
        std_score = float(np.std(outer_scores, ddof=1)) if len(outer_scores) > 1 else 0.0
        append_log(
            log_path,
            f"SUMMARY run={run_id} selected_k_by_outer_fold={selected_counts} "
            f"outer_macro_f1_mean={mean_score:.6f} "
            f"outer_macro_f1_sample_sd={std_score:.6f} "
            f"training_only_selected_k={final_search['best_k']} "
            f"training_only_selection_macro_f1={final_search['best_macro_f1']:.6f} "
            "NOTE=selection_score_is_not_an_independent_test_estimate",
        )
        append_log(
            log_path,
            f"END run={run_id} total_elapsed_seconds={time.perf_counter() - started:.2f} "
            "official_sat.tst=NOT_ACCESSED",
        )
        print(f"\nAppended full run log to: {log_path}")
    except Exception:
        append_log(log_path, f"FAILED run={run_id}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
