import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph import build_corr_graph
from landsat_data import ROOT, load_landsat, standardize
from mi import relevance_scores
from pca import PCA
from pcap_data import load_pcap, load_splits
from selection import select_features, select_naive

RESULTS_DIR = ROOT / "results"
TRIALS = 7
SAMPLE_COUNT = 500
THRESHOLD = 0.7
FEATURE_COUNTS = (50, 100, 200, 400, 800, 1600)
TIMING_COLUMNS = ("corr_s", "graph_s", "heap_bfs_select_s", "heap_bfs_total_s", "naive_s")
STAGE_COLUMNS = ("mi_s", "corr_s", "graph_s", "heap_bfs_select_s", "naive_s", "pca_s")
DATA_SCALING = {
    "landsat": {"rows": (500, 1000, 2000, 4435), "threshold": 0.9, "trials": 7},
    "pcap": {"rows": (1000, 3000, 10000, 30000, 100000, 300000, 591484), "threshold": 0.84, "trials": 3},
}


def median_time(function, *args, trials=TRIALS):
    timings, result = [], None
    for _ in range(trials):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)
    return result, float(np.median(timings))


def make_data(feature_count, rng, sample_count=SAMPLE_COUNT):
    latent = rng.normal(size=(sample_count, feature_count // 4 + 1))
    X = np.column_stack([latent[:, j // 4] + 0.5 * rng.normal(size=sample_count)
                         for j in range(feature_count)])
    return X, list(rng.random(feature_count))


def time_one_size(d, rng, threshold, trials):
    X, scores = make_data(d, rng)
    C, corr_s = median_time(lambda: np.abs(np.corrcoef(X.T)), trials=trials)
    graph, graph_s = median_time(build_corr_graph, C, threshold, trials=trials)
    selected, selection_s = median_time(select_features, scores, graph, trials=trials)
    naive, naive_s = median_time(select_naive, scores, C, threshold, trials=trials)
    if sorted(selected) != sorted(naive):
        raise AssertionError(f"Selection mismatch at d={d}")
    return {
        "d": d, "trials": trials,
        "edges": sum(len(neighbours) for neighbours in graph.adj) // 2,
        "kept": len(selected),
        "corr_s": corr_s, "graph_s": graph_s, "heap_bfs_select_s": selection_s,
        "heap_bfs_total_s": graph_s + selection_s, "naive_s": naive_s,
    }


def growth_exponents(results, min_d=200, size_column="d", columns=TIMING_COLUMNS):
    subset = results[results[size_column] >= min_d]
    exponents = {}
    if len(subset) < 2:
        return exponents
    for column in columns:
        times = subset[column].to_numpy()
        if np.all(times > 0):
            exponents[column] = float(np.polyfit(np.log(subset[size_column].to_numpy()), np.log(times), 1)[0])
    return exponents


def plot_runtimes(results, trials=TRIALS):
    fig, ax = plt.subplots()
    for column in TIMING_COLUMNS:
        ax.loglog(results["d"], results[column], marker="o", label=column)
    ax.set_xlabel("Number of features d")
    ax.set_ylabel("Median runtime (seconds)")
    ax.set_title(f"Runtime vs number of features ({trials} trials)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return fig


def run_complexity(feature_counts=FEATURE_COUNTS, threshold=THRESHOLD, trials=TRIALS, save=False, seed=0):
    rng = np.random.default_rng(seed)
    results = pd.DataFrame([time_one_size(d, rng, threshold, trials) for d in feature_counts])
    fig = plot_runtimes(results, trials)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(RESULTS_DIR / "complexity.csv", index=False)
        fig.savefig(RESULTS_DIR / "complexity.png", dpi=200, bbox_inches="tight")
    return results, growth_exponents(results), fig


def load_scaling_data(dataset):
    if dataset == "landsat":
        X, y, _, _ = load_landsat()
        (X,) = standardize(X)
        return X, y
    if dataset == "pcap":
        data = load_pcap()
        splits = load_splits(n_rows=len(data.y))
        X = data.X[splits.features].to_numpy(dtype=float)[splits.dev_idx]
        X = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        (X,) = standardize(X)
        return X, data.y[splits.dev_idx]
    raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATA_SCALING)}")


def correlation_matrix(X):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)


def time_stages(X, y, threshold, trials=TRIALS):
    scores, mi_s = median_time(relevance_scores, X, y, trials=trials)
    C, corr_s = median_time(correlation_matrix, X, trials=trials)
    graph, graph_s = median_time(build_corr_graph, C, threshold, trials=trials)
    selected, selection_s = median_time(select_features, scores, graph, trials=trials)
    naive, naive_s = median_time(select_naive, scores, C, threshold, trials=trials)
    if sorted(selected) != sorted(naive):
        raise AssertionError(f"Selection mismatch at n={len(y)}")
    _, pca_s = median_time(lambda: PCA(var_ratio=0.95).fit(X), trials=trials)
    return {"n": len(y), "d": X.shape[1], "trials": trials,
            "edges": sum(len(neighbours) for neighbours in graph.adj) // 2, "kept": len(selected),
            "mi_s": mi_s, "corr_s": corr_s, "graph_s": graph_s, "heap_bfs_select_s": selection_s,
            "naive_s": naive_s, "pca_s": pca_s}


def plot_stage_times(results, dataset):
    fig, ax = plt.subplots()
    for column in STAGE_COLUMNS:
        ax.loglog(results["n"], results[column], marker="o", label=column)
    ax.set_xlabel("Number of rows n")
    ax.set_ylabel("Median runtime (seconds)")
    ax.set_title(f"{dataset}: runtime of each stage vs number of rows")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return fig


def run_data_scaling(dataset="pcap", rows=None, threshold=None, trials=None, save=False, seed=0, show=print):
    settings = DATA_SCALING.get(dataset)
    if settings is None:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATA_SCALING)}")
    rows = rows or settings["rows"]
    threshold = settings["threshold"] if threshold is None else threshold
    trials = trials or settings["trials"]
    X, y = load_scaling_data(dataset)
    rng = np.random.default_rng(seed)
    records = []
    for n in rows:
        n = min(n, len(y))
        picked = np.sort(rng.choice(len(y), size=n, replace=False))
        records.append(time_stages(X[picked], y[picked], threshold, trials))
        show(f"{dataset} n={n:,}: MI {records[-1]['mi_s']:.3f}s, heap+BFS {records[-1]['heap_bfs_select_s']:.5f}s")
    results = pd.DataFrame(records)
    exponents = growth_exponents(results, min_d=results["n"].iloc[len(results) // 2], size_column="n",
                                 columns=STAGE_COLUMNS)
    fig = plot_stage_times(results, dataset)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(RESULTS_DIR / f"complexity_rows_{dataset}.csv", index=False)
        fig.savefig(RESULTS_DIR / f"complexity_rows_{dataset}.png", dpi=200, bbox_inches="tight")
    return results, exponents, fig
