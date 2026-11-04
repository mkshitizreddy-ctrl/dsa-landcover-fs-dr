"""Recheck V4 feature stability and paired full-feature baselines on sat.trn.

This script reads only V4 summary masks and sat.trn. It replays the same
stratified splits for a full-feature custom-kNN baseline and writes separate
v5 analysis files without changing the V4 source CSVs.
"""

import csv
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_loader import DatasetConfig, load_dataset
from knn import KNN
from validation_pipeline import make_validation_split

SOURCE = ROOT / "results" / "pso_summary_v4.csv"
PAIRED = ROOT / "results" / "pso_paired_baseline_v5.csv"
FREQUENCY = ROOT / "results" / "pso_feature_frequency_v5.csv"
JACCARD = ROOT / "results" / "pso_pairwise_jaccard_v5.csv"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError(f"Refusing to append incompatible schema to {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def metric_row(y, pred):
    return (
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, average="macro", zero_division=0)),
        float(balanced_accuracy_score(y, pred)),
    )


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError("V4 summary contains no runs")
    seeds = [int(row["seed"]) for row in source_rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Expected one V4 summary row per distinct seed")

    train_path = ROOT / "data" / "sat.trn"
    dataset = load_dataset(DatasetConfig(path=str(train_path), delimiter=r"\s+", header=None, target_column=-1))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selected_by_seed = {}
    paired_rows = []
    for row in source_rows:
        seed = int(row["seed"])
        selected = sorted({int(value) for value in row["selected_feature_indices"].split(";") if value})
        if not selected or min(selected) < 0 or max(selected) >= dataset.X.shape[1]:
            raise ValueError(f"Invalid feature mask for seed {seed}")
        selected_by_seed[seed] = set(selected)
        split = make_validation_split(dataset, validation_fraction=float(row["validation_fraction"]), seed=seed, scale=True)
        baseline = KNN(k=int(row["k_neighbors"])).fit(split.X_train, split.y_train)
        baseline_metrics = metric_row(split.y_valid, baseline.predict(split.X_valid))
        pso_metrics = tuple(float(row[key]) for key in ("accuracy", "macro_f1", "balanced_accuracy"))
        paired_rows.append({
            "analysis_run_id": run_id, "seed": seed,
            "baseline_accuracy": baseline_metrics[0], "pso_accuracy": pso_metrics[0],
            "accuracy_delta_pso_minus_baseline": pso_metrics[0] - baseline_metrics[0],
            "baseline_macro_f1": baseline_metrics[1], "pso_macro_f1": pso_metrics[1],
            "macro_f1_delta_pso_minus_baseline": pso_metrics[1] - baseline_metrics[1],
            "baseline_balanced_accuracy": baseline_metrics[2], "pso_balanced_accuracy": pso_metrics[2],
            "balanced_accuracy_delta_pso_minus_baseline": pso_metrics[2] - baseline_metrics[2],
            "pso_features": len(selected), "total_features": dataset.X.shape[1],
        })

    frequency_rows = []
    for feature in range(dataset.X.shape[1]):
        count = sum(feature in subset for subset in selected_by_seed.values())
        frequency_rows.append({
            "analysis_run_id": run_id, "feature_index_zero_based": feature,
            "selected_runs": count, "total_runs": len(selected_by_seed),
            "selection_frequency": count / len(selected_by_seed),
        })
    jaccard_rows = []
    for left, right in combinations(sorted(selected_by_seed), 2):
        a, b = selected_by_seed[left], selected_by_seed[right]
        jaccard_rows.append({
            "analysis_run_id": run_id, "seed_a": left, "seed_b": right,
            "intersection_size": len(a & b), "union_size": len(a | b),
            "jaccard_similarity": len(a & b) / len(a | b),
        })

    write_csv(PAIRED, paired_rows, list(paired_rows[0]))
    write_csv(FREQUENCY, frequency_rows, list(frequency_rows[0]))
    if jaccard_rows:
        write_csv(JACCARD, jaccard_rows, list(jaccard_rows[0]))
    delta = np.asarray([row["macro_f1_delta_pso_minus_baseline"] for row in paired_rows])
    jac = np.asarray([row["jaccard_similarity"] for row in jaccard_rows]) if jaccard_rows else np.array([])
    print(f"Runs: {len(paired_rows)}; mean paired macro-F1 delta: {delta.mean():+.6f}; better splits: {(delta > 0).sum()}/{len(delta)}")
    print(f"Mean pairwise subset Jaccard: {jac.mean():.6f}; pairwise range: {jac.min():.6f}–{jac.max():.6f}")
    print(f"Wrote: {PAIRED}\nWrote: {FREQUENCY}\nWrote: {JACCARD}")


if __name__ == "__main__":
    main()
