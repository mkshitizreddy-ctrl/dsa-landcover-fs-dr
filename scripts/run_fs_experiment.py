import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from graph import build_corr_graph
from knn import KNN
from landsat_data import ROOT, load_landsat, standardize
from mi import relevance_scores
from selection import select_features

THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
RESULTS_PATH = ROOT / "results" / "fs_experiment.csv"


def evaluate_knn(name, cols, X_train, y_train, X_test, y_test, k=5):
    start = time.time()
    model = KNN(k).fit(X_train[:, cols], y_train)
    built = time.time()
    pred = model.predict(X_test[:, cols])
    done = time.time()
    return {
        "setting": name,
        "features": len(cols),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "build_s": round(built - start, 2),
        "query_s": round(done - built, 2),
    }


def sklearn_knn_accuracy(X_train, y_train, X_test, y_test, k=5):
    pred = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)
    return round(accuracy_score(y_test, pred), 4)


def run_fs_experiment(thresholds=THRESHOLDS, save=False):
    X_train, y_train, X_test, y_test = load_landsat()
    X_train, X_test = standardize(X_train, X_test)
    scores = relevance_scores(X_train, y_train)
    C = np.abs(np.corrcoef(X_train.T))
    data = (X_train, y_train, X_test, y_test)

    rows = [evaluate_knn(f"raw (all {X_train.shape[1]})", list(range(X_train.shape[1])), *data)]
    for t in thresholds:
        selected = sorted(select_features(scores, build_corr_graph(C, t)))
        rows.append(evaluate_knn(f"heap+BFS t={t}", selected, *data))
    results = pd.DataFrame(rows)
    if save:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(RESULTS_PATH, index=False)
    return results, sklearn_knn_accuracy(*data)
