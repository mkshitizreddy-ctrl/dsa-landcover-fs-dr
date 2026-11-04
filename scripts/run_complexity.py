import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph import build_corr_graph
from landsat_data import ROOT
from selection import select_features, select_naive

RESULTS_DIR = ROOT / "results"
TRIALS = 7
SAMPLE_COUNT = 500
THRESHOLD = 0.7
FEATURE_COUNTS = (50, 100, 200, 400, 800, 1600)
TIMING_COLUMNS = ("corr_s", "graph_s", "heap_bfs_select_s", "heap_bfs_total_s", "naive_s")


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


def growth_exponents(results, min_d=200):
    subset = results[results["d"] >= min_d]
    exponents = {}
    if len(subset) < 2:
        return exponents
    for column in TIMING_COLUMNS:
        times = subset[column].to_numpy()
        if np.all(times > 0):
            exponents[column] = float(np.polyfit(np.log(subset["d"].to_numpy()), np.log(times), 1)[0])
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
