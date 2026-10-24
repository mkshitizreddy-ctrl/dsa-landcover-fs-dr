
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

sys.path.insert(0, SRC_DIR)

from graph import build_corr_graph
from selection import select_features, select_naive


TRIALS = 7
SAMPLE_COUNT = 500
THRESHOLD = 0.7
FEATURE_COUNTS = (50, 100, 200, 400, 800, 1600)


def timed(function, *args):
    """Run one timed call and return its result and elapsed seconds."""
    start = time.perf_counter()
    result = function(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def median_time(function, *args, trials=TRIALS):
    """Repeat a function and return its final result and median runtime."""
    timings = []
    result = None

    for _ in range(trials):
        result, elapsed = timed(function, *args)
        timings.append(elapsed)

    return result, float(np.median(timings))


def make_data(feature_count, rng):
    """Create synthetic features with groups of four correlated features."""
    latent_count = feature_count // 4 + 1
    latent = rng.normal(size=(SAMPLE_COUNT, latent_count))

    X = np.column_stack([
        latent[:, j // 4] + 0.5 * rng.normal(size=SAMPLE_COUNT)
        for j in range(feature_count)
    ])

    scores = list(rng.random(feature_count))
    return X, scores


def main():
    rng = np.random.default_rng(0)
    rows = []

    for d in FEATURE_COUNTS:
        print(f"Benchmarking d={d} features ({TRIALS} trials)...")

        X, scores = make_data(d, rng)

        # Correlation computation: O(m * d^2).
        C, corr_s = median_time(
            lambda: np.abs(np.corrcoef(X.T))
        )

        # Ensure the matrix is symmetric with a zero diagonal.
        C = np.triu(C) + np.triu(C, 1).T

        # Graph construction: O(d^2).
        graph, graph_s = median_time(
            build_corr_graph, C, THRESHOLD
        )

        # Heap+BFS feature selection.
        selected, selection_s = median_time(
            select_features, scores, graph
        )

        # Naive selection baseline.
        naive, naive_s = median_time(
            select_naive, scores, C, THRESHOLD
        )

        # Confirm the implementations agree on this input.
        assert sorted(selected) == sorted(naive), (
            f"Selection mismatch at d={d}"
        )

        rows.append({
            "d": d,
            "trials": TRIALS,
            "edges": sum(len(neighbors) for neighbors in graph.adj) // 2,
            "kept": len(selected),
            "corr_s": corr_s,
            "graph_s": graph_s,
            "heap_bfs_select_s": selection_s,
            "heap_bfs_total_s": graph_s + selection_s,
            "naive_s": naive_s,
        })

    results = pd.DataFrame(rows)

    print("\nMedian runtime results:")
    print(results.round(5).to_string(index=False))

    # Empirical growth exponents fitted to median timings for d >= 200.
    print("\nEmpirical growth exponent (time ~ d^k), fitted on d >= 200:")
    subset = results[results["d"] >= 200]

    timing_columns = (
        "corr_s",
        "graph_s",
        "heap_bfs_select_s",
        "heap_bfs_total_s",
        "naive_s",
    )

    for column in timing_columns:
        times = subset[column].to_numpy()

        if np.any(times <= 0):
            print(f"  {column}: skipped because a timing was non-positive")
            continue

        exponent = np.polyfit(
            np.log(subset["d"].to_numpy()),
            np.log(times),
            1,
        )[0]

        print(f"  {column}: k = {exponent:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot the measured median runtimes.
    plot_columns = (
        "corr_s",
        "graph_s",
        "heap_bfs_select_s",
        "heap_bfs_total_s",
        "naive_s",
    )

    plt.figure()

    for column in plot_columns:
        plt.loglog(
            results["d"],
            results[column],
            marker="o",
            label=column,
        )

    plt.xlabel("Number of features d")
    plt.ylabel("Median runtime (seconds)")
    plt.title(f"Runtime vs number of features ({TRIALS} trials)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)

    plot_path = os.path.join(RESULTS_DIR, "complexity.png")
    csv_path = os.path.join(RESULTS_DIR, "complexity.csv")

    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    results.to_csv(csv_path, index=False)

    print(f"\nMedian results saved to: {csv_path}")
    print(f"Runtime plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
