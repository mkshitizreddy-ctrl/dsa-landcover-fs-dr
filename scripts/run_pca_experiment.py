
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from mi import relevance_scores
from graph import build_corr_graph
from selection import select_features
from pca import PCA
from knn import KNN


BASE_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/statlog/satimage/"
)


def load(name):
    """Load a local dataset file, downloading it only if it is missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)

    if os.path.isfile(path):
        print(f"Using local dataset: {path}")
    else:
        url = BASE_URL + name
        print(f"Local file missing; downloading {name} from UCI...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved dataset to: {path}")

    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1].astype(int)


def prepare_features(Xtr, Xte, ytr, setting, threshold=None):
    """Prepare the feature representation using training data only."""
    if setting == "Raw":
        return Xtr, Xte

    if setting == "FS":
        scores = relevance_scores(Xtr, ytr)
        correlation = np.abs(np.corrcoef(Xtr.T))
        graph = build_corr_graph(correlation, threshold)
        selected = sorted(select_features(scores, graph))

        return Xtr[:, selected], Xte[:, selected]

    if setting == "PCA":
        pca = PCA(0.95).fit(Xtr)
        return pca.transform(Xtr), pca.transform(Xte)

    if setting == "FS+PCA":
        scores = relevance_scores(Xtr, ytr)
        correlation = np.abs(np.corrcoef(Xtr.T))
        graph = build_corr_graph(correlation, threshold)
        selected = sorted(select_features(scores, graph))

        Xtr_selected = Xtr[:, selected]
        Xte_selected = Xte[:, selected]

        pca = PCA(0.95).fit(Xtr_selected)
        return (
            pca.transform(Xtr_selected),
            pca.transform(Xte_selected),
        )

    raise ValueError(f"Unknown setting: {setting}")


def run(label, Xtr, Xte, ytr, yte, setting, threshold=None,
        shared_preprocessing_s=0.0):
    """Measure feature preparation, model fitting, and prediction separately."""
    build_start = time.perf_counter()

    Xtr_used, Xte_used = prepare_features(
        Xtr, Xte, ytr, setting, threshold
    )

    model = KNN(5).fit(Xtr_used, ytr)

    build_s = time.perf_counter() - build_start

    prediction_start = time.perf_counter()
    predictions = model.predict(Xte_used)
    predict_s = time.perf_counter() - prediction_start

    total_s = shared_preprocessing_s + build_s + predict_s

    return {
        "setting": label,
        "dims": Xtr_used.shape[1],
        "accuracy": round(accuracy_score(yte, predictions), 4),
        "macro_f1": round(
            f1_score(yte, predictions, average="macro"), 4
        ),
        "build_s": round(build_s, 4),
        "predict_s": round(predict_s, 4),
        "total_s": round(total_s, 4),
    }


def main(synthetic=False):
    if synthetic:
        rng = np.random.default_rng(0)

        Xtr = rng.standard_normal((500, 12))
        Xtr[:, 5] = Xtr[:, 0] + 0.1 * rng.standard_normal(500)
        ytr = (Xtr[:, 0] + Xtr[:, 1] > 0).astype(int) + 1

        Xte = rng.standard_normal((100, 12))
        Xte[:, 5] = Xte[:, 0] + 0.1 * rng.standard_normal(100)
        yte = (Xte[:, 0] + Xte[:, 1] > 0).astype(int) + 1
    else:
        Xtr, ytr = load("sat.trn")
        Xte, yte = load("sat.tst")

    print(f"Training samples: {len(ytr)}")
    print(f"Test samples: {len(yte)}")

    # Standardize using training statistics only.
    scaling_start = time.perf_counter()

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0

    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    scaling_s = time.perf_counter() - scaling_start
    print(f"Shared standardization time: {scaling_s:.4f} seconds")

    settings = [
        ("Raw", "Raw", None),
        ("FS only (t=0.95)", "FS", 0.95),
        ("FS only (t=0.9)", "FS", 0.9),
        ("PCA only (95% var)", "PCA", None),
        ("FS (t=0.95) + PCA", "FS+PCA", 0.95),
        ("FS (t=0.9) + PCA", "FS+PCA", 0.9),
    ]

    rows = []

    for label, setting, threshold in settings:
        rows.append(
            run(
                label,
                Xtr,
                Xte,
                ytr,
                yte,
                setting,
                threshold,
                shared_preprocessing_s=scaling_s,
            )
        )

    results = pd.DataFrame(rows)
    print("\nExperiment results:")
    print(results.to_string(index=False))

    if not synthetic:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, "pca_experiment.csv")
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main(synthetic="--synthetic" in sys.argv)
