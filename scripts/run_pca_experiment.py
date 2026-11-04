import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from graph import build_corr_graph
from knn import KNN
from landsat_data import ROOT, load_landsat, standardize
from mi import relevance_scores
from pca import PCA
from selection import select_features

RESULTS_PATH = ROOT / "results" / "pca_experiment.csv"
SETTINGS = (
    ("Raw", "Raw", None),
    ("FS only (t=0.95)", "FS", 0.95),
    ("FS only (t=0.9)", "FS", 0.9),
    ("PCA only (95% var)", "PCA", None),
    ("FS (t=0.95) + PCA", "FS+PCA", 0.95),
    ("FS (t=0.9) + PCA", "FS+PCA", 0.9),
)


def make_synthetic_data(seed=0):
    rng = np.random.default_rng(seed)

    def sample(n):
        X = rng.standard_normal((n, 12))
        X[:, 5] = X[:, 0] + 0.1 * rng.standard_normal(n)
        y = (X[:, 0] + X[:, 1] > 0).astype(int) + 1
        return X, y

    X_train, y_train = sample(500)
    X_test, y_test = sample(100)
    return X_train, y_train, X_test, y_test


def select_columns(X_train, y_train, threshold):
    scores = relevance_scores(X_train, y_train)
    graph = build_corr_graph(np.abs(np.corrcoef(X_train.T)), threshold)
    return sorted(select_features(scores, graph))


def prepare_features(X_train, X_test, y_train, setting, threshold=None):
    if setting == "Raw":
        return X_train, X_test
    if setting in ("FS", "FS+PCA"):
        cols = select_columns(X_train, y_train, threshold)
        X_train, X_test = X_train[:, cols], X_test[:, cols]
        if setting == "FS":
            return X_train, X_test
    if setting in ("PCA", "FS+PCA"):
        pca = PCA(0.95).fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    raise ValueError(f"Unknown setting: {setting}")


def evaluate_setting(label, X_train, X_test, y_train, y_test, setting, threshold=None, scaling_s=0.0):
    start = time.perf_counter()
    X_train_used, X_test_used = prepare_features(X_train, X_test, y_train, setting, threshold)
    model = KNN(5).fit(X_train_used, y_train)
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test_used)
    predict_s = time.perf_counter() - start
    return {
        "setting": label,
        "dims": X_train_used.shape[1],
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "build_s": round(build_s, 4),
        "predict_s": round(predict_s, 4),
        "total_s": round(scaling_s + build_s + predict_s, 4),
    }


def run_pca_experiment(synthetic=False, save=False):
    X_train, y_train, X_test, y_test = make_synthetic_data() if synthetic else load_landsat()
    start = time.perf_counter()
    X_train, X_test = standardize(X_train, X_test)
    scaling_s = time.perf_counter() - start
    rows = [evaluate_setting(label, X_train, X_test, y_train, y_test, setting, threshold, scaling_s)
            for label, setting, threshold in SETTINGS]
    results = pd.DataFrame(rows)
    if save and not synthetic:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(RESULTS_PATH, index=False)
    return results
