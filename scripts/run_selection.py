import numpy as np
import pandas as pd

from graph import build_corr_graph, components
from landsat_data import load_landsat, standardize
from mi import relevance_scores
from selection import select_features, select_naive, select_per_component
from sorting import merge_sort

THRESHOLDS = (0.7, 0.8, 0.9, 0.95)


def top_features(scores, n=5):
    return merge_sort(list(range(len(scores))), key=lambda j: scores[j], reverse=True)[:n]


def compare_selection_methods(X_train, y_train, thresholds=THRESHOLDS):
    scores = relevance_scores(X_train, y_train)
    C = np.abs(np.corrcoef(X_train.T))
    rows = []
    for t in thresholds:
        graph = build_corr_graph(C, t)
        selected = select_features(scores, graph)
        rows.append({
            "threshold": t,
            "components": components(graph)[1],
            "heap_bfs": len(selected),
            "per_component": len(select_per_component(scores, graph)),
            "same_as_naive": sorted(selected) == sorted(select_naive(scores, C, t)),
        })
    return pd.DataFrame(rows), top_features(scores)


def run_selection(thresholds=THRESHOLDS):
    X_train, y_train, _, _ = load_landsat()
    (X_train,) = standardize(X_train)
    return compare_selection_methods(X_train, y_train, thresholds)
