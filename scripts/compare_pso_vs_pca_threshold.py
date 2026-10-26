"""Fair nested-CV comparison of PSO k selection and 95% variance threshold.

Both methods use the same training-only SVD, stratified outer folds, inner
training split, scikit-learn kNN, and held-out outer-fold scoring. sat.tst is
never loaded. Results append to results/pso_vs_threshold_runs.txt.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_loader import DatasetConfig, load_dataset
from run_pso_pca_svd import (
    append_log,
    fit_svd_projection,
    metrics,
    optimize_component_count,
)
from validation_pipeline import make_validation_split


def threshold_count(singular_values, threshold):
    explained = np.square(singular_values)
    cumulative = np.cumsum(explained) / np.sum(explained)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--log-path", type=Path,
        default=ROOT / "results" / "pso_vs_threshold_runs.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.population < 5:
        raise ValueError("folds >= 2, epochs >= 1, and Mealpy population >= 5 are required")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")
    if not 0 < args.variance_threshold <= 1:
        raise ValueError("variance-threshold must be in (0, 1]")
    if args.k_neighbors < 1:
        raise ValueError("k-neighbors must be positive")

    log_path = args.log_path
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started = time.perf_counter()
    data_path = ROOT / "data" / "sat.trn"
    dataset = load_dataset(DatasetConfig(
        path=str(data_path), delimiter=r"\s+", header=None,
        target_column=-1, task="classification",
    ))
    append_log(
        log_path,
        f"START run={run_id} rows={len(dataset.y)} features={dataset.X.shape[1]} "
        f"folds={args.folds} seed={args.seed} variance_threshold={args.variance_threshold} "
        f"PSO_epochs={args.epochs} PSO_population={args.population} "
        f"classifier=sklearn.KNeighborsClassifier(k={args.k_neighbors}) "
        "same_outer_folds=same inner training data=SVD fit on training only "
        "official_sat.tst=NOT_ACCESSED",
    )

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    records = []
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

        # Both component-count rules see the same inner training matrix.
        _, _, inner_singular_values = fit_svd_projection(
            inner.X_train, inner.X_valid
        )
        threshold_k = threshold_count(
            inner_singular_values, args.variance_threshold
        )
        pso = optimize_component_count(
            inner.X_train, inner.y_train, inner.X_valid, inner.y_valid,
            seed=args.seed + fold, epochs=args.epochs,
            population=args.population, k_neighbors=args.k_neighbors,
            n_jobs=args.n_jobs, log_path=log_path, run_id=run_id,
            label=f"fold_{fold}_pso_search",
        )

        # Refit SVD/scaling on outer-train once and score both methods on the
        # exact same untouched validation rows.
        outer_train_scores, outer_valid_scores, _ = fit_svd_projection(
            outer_X_train, outer_X_valid
        )
        fold_record = {"threshold": [], "pso": []}
        for method, component_count in (
            ("threshold", threshold_k), ("pso", pso["best_k"])
        ):
            classifier = KNeighborsClassifier(
                n_neighbors=args.k_neighbors, n_jobs=args.n_jobs
            )
            classifier.fit(outer_train_scores[:, :component_count], outer_y_train)
            prediction = classifier.predict(outer_valid_scores[:, :component_count])
            result = metrics(outer_y_valid, prediction)
            fold_record[method] = {
                "components": component_count,
                **result,
            }
        records.append(fold_record)
        append_log(
            log_path,
            f"FOLD_RESULT run={run_id} fold={fold}/{args.folds} "
            f"threshold_k={threshold_k} "
            f"threshold_macro_f1={fold_record['threshold']['macro_f1']:.6f} "
            f"pso_k={pso['best_k']} "
            f"pso_macro_f1={fold_record['pso']['macro_f1']:.6f} "
            f"paired_macro_f1_delta="
            f"{fold_record['pso']['macro_f1'] - fold_record['threshold']['macro_f1']:+.6f} "
            f"threshold_accuracy={fold_record['threshold']['accuracy']:.6f} "
            f"pso_accuracy={fold_record['pso']['accuracy']:.6f} "
            f"elapsed_seconds={time.perf_counter() - fold_started:.2f}",
        )

    threshold_scores = [row["threshold"]["macro_f1"] for row in records]
    pso_scores = [row["pso"]["macro_f1"] for row in records]
    deltas = np.asarray(pso_scores) - np.asarray(threshold_scores)
    threshold_counts = [row["threshold"]["components"] for row in records]
    pso_counts = [row["pso"]["components"] for row in records]

    # Also record the training-only component counts on all sat.trn. These are
    # choices for a later fit, not additional performance estimates.
    full_inner = make_validation_split(
        dataset, validation_fraction=args.validation_fraction,
        seed=args.seed, scale=False,
    )
    _, _, full_singular_values = fit_svd_projection(
        full_inner.X_train, full_inner.X_valid
    )
    final_threshold_k = threshold_count(
        full_singular_values, args.variance_threshold
    )
    final_pso = optimize_component_count(
        full_inner.X_train, full_inner.y_train,
        full_inner.X_valid, full_inner.y_valid,
        seed=args.seed, epochs=args.epochs, population=args.population,
        k_neighbors=args.k_neighbors, n_jobs=args.n_jobs,
        log_path=log_path, run_id=run_id, label="full_training_pso_search",
    )

    append_log(
        log_path,
        f"SUMMARY run={run_id} threshold_k_by_fold={threshold_counts} "
        f"pso_k_by_fold={pso_counts} "
        f"threshold_macro_f1_mean={np.mean(threshold_scores):.6f} "
        f"threshold_macro_f1_sample_sd={np.std(threshold_scores, ddof=1):.6f} "
        f"pso_macro_f1_mean={np.mean(pso_scores):.6f} "
        f"pso_macro_f1_sample_sd={np.std(pso_scores, ddof=1):.6f} "
        f"paired_delta_mean={np.mean(deltas):+.6f} "
        f"PSO_wins={int(np.sum(deltas > 0))}/{args.folds} "
        f"full_training_threshold_k={final_threshold_k} "
        f"full_training_PSO_k={final_pso['best_k']} "
        "NOTE=outer-fold_metrics_are_comparable; final_inner_scores_are_for_selection_only",
    )
    append_log(
        log_path,
        f"END run={run_id} total_elapsed_seconds={time.perf_counter() - started:.2f} "
        "official_sat.tst=NOT_ACCESSED",
    )
    print(f"\nAppended comparison to: {log_path}")


if __name__ == "__main__":
    main()
