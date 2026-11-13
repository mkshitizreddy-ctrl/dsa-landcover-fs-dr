import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from graph import build_corr_graph, components
from knn import KNN
from landsat_data import load_landsat, standardize
from pca import PCA
from pcap_feature_selection import feature_statistics
from selection import select_features, select_naive, select_per_component

METHODS = {"Heap + BFS": "heap_bfs", "Best feature per component (Union-Find)": "per_component",
           "Naive greedy (sorted list)": "naive"}


def feature_names(n_features=36):
    return [f"pixel {j // 4 + 1}, band {j % 4 + 1}" for j in range(n_features)]


def validation_split(validation_fraction=0.2, seed=42):
    X, y, _, _ = load_landsat()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_fraction, stratify=y,
                                                          random_state=seed)
    X_train, X_valid = standardize(X_train, X_valid)
    return X_train, X_valid, y_train, y_valid


def explore_statistics(X_train, y_train):
    return feature_statistics(X_train, y_train)


def select(scores, C, t, method="heap_bfs"):
    g = build_corr_graph(C, t)
    if method == "heap_bfs":
        chosen = select_features(scores, g)
    elif method == "per_component":
        chosen = select_per_component(scores, g)
    elif method == "naive":
        chosen = select_naive(scores, C, t)
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS.values())}")
    _, n_components = components(g)
    return {"features": sorted(chosen), "edges": sum(len(a) for a in g.adj) // 2, "components": n_components}


def threshold_sweep(scores, C, thresholds, method="heap_bfs"):
    rows = []
    for t in thresholds:
        result = select(scores, C, t, method)
        rows.append({"threshold": round(float(t), 2), "features_kept": len(result["features"]),
                     "edges": result["edges"], "components": result["components"]})
    return pd.DataFrame(rows)


def evaluate(X_train, y_train, X_valid, y_valid, columns, k=5, pca_ratio=None):
    A_train, A_valid = X_train[:, columns], X_valid[:, columns]
    if pca_ratio:
        pca = PCA(var_ratio=pca_ratio).fit(A_train)
        A_train, A_valid = pca.transform(A_train), pca.transform(A_valid)
    model = KNN(k).fit(A_train, y_train)
    start = time.perf_counter()
    pred = model.predict(A_valid)
    seconds = time.perf_counter() - start
    return {"dimensions": A_train.shape[1], "accuracy": accuracy_score(y_valid, pred),
            "macro_f1": f1_score(y_valid, pred, average="macro"), "query_seconds": seconds}


def score_table(scores, selected):
    names = feature_names(len(scores))
    table = pd.DataFrame({"feature": names, "mutual_information": np.asarray(scores, dtype=float)})
    table["status"] = np.where(np.isin(np.arange(len(scores)), selected), "kept", "removed")
    return table.sort_values("mutual_information", ascending=False, kind="stable").reset_index(drop=True)
