from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LANDSAT_MH_PATH = ROOT / "results" / "mh_comparison_v1.csv"
ALGORITHMS = ("GA", "DE", "PSO", "GWO", "WOA", "HHO", "ABC", "MFO", "SSA", "SMA")
LANDSAT_TASKS = ("fs", "threshold", "components", "joint")
PCAP_SEARCH_PATHS = {"grid": ROOT / "results" / "pcap_mh_v1" / "runs.csv",
                     "selection": ROOT / "results" / "pcap_fs_search_v1" / "runs.csv"}
RANDOM = "RANDOM"
STATS_DIR = ROOT / "results" / "stats_v1"


def score_matrix(runs, block_columns, algorithm_column, score_column, algorithms=None):
    if algorithms is not None:
        runs = runs[runs[algorithm_column].isin(algorithms)]
    table = runs.pivot_table(index=block_columns, columns=algorithm_column, values=score_column, aggfunc="mean")
    if table.empty or table.isna().any().any():
        raise ValueError("Every algorithm needs a score in every block")
    return table


def average_ranks(table, higher_is_better=True):
    scores = -table if higher_is_better else table
    return scores.rank(axis=1, method="average").mean().sort_values()


def friedman_test(table):
    if table.shape[1] < 3:
        raise ValueError("The Friedman test needs at least three algorithms")
    if table.shape[0] < 2:
        raise ValueError("The Friedman test needs at least two blocks")
    statistic, p_value = stats.friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    return {"n_blocks": int(table.shape[0]), "n_algorithms": int(table.shape[1]),
            "statistic": float(statistic), "p_value": float(p_value)}


def nemenyi_critical_difference(n_algorithms, n_blocks, alpha=0.05):
    if n_algorithms < 2 or n_blocks < 1:
        raise ValueError("The Nemenyi test needs at least two algorithms and one block")
    q = stats.studentized_range.ppf(1 - alpha, n_algorithms, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_blocks)))


def nemenyi_pairs(table, alpha=0.05, higher_is_better=True):
    ranks = average_ranks(table, higher_is_better)
    cd = nemenyi_critical_difference(table.shape[1], table.shape[0], alpha)
    names = list(ranks.index)
    rows = []
    for i, better in enumerate(names):
        for worse in names[i + 1:]:
            gap = float(ranks[worse] - ranks[better])
            rows.append({"better": better, "worse": worse, "rank_better": float(ranks[better]),
                         "rank_worse": float(ranks[worse]), "rank_gap": gap, "critical_difference": cd,
                         "significant": gap > cd})
    return pd.DataFrame(rows)


def landsat_table(task, path=LANDSAT_MH_PATH, score_column="macro_f1"):
    runs = pd.read_csv(path)
    return score_matrix(runs[runs.task == task], ["seed", "outer_fold"], "method", score_column, ALGORITHMS)


def landsat_nemenyi(task, path=LANDSAT_MH_PATH, score_column="macro_f1", alpha=0.05):
    return nemenyi_pairs(landsat_table(task, path, score_column), alpha)


def landsat_friedman(path=LANDSAT_MH_PATH, score_column="macro_f1", tasks=LANDSAT_TASKS):
    rows = []
    for task in tasks:
        table = landsat_table(task, path, score_column)
        ranks = average_ranks(table)
        rows.append({"task": task, **friedman_test(table),
                     "best_algorithm": ranks.index[0], "best_mean_rank": float(ranks.iloc[0])})
    return pd.DataFrame(rows)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for position, index in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - position) * p[index]))
        adjusted[index] = running
    return adjusted


def kruskal_test(groups):
    if len(groups) < 3:
        raise ValueError("The Kruskal-Wallis test needs at least three algorithms")
    statistic, p_value = stats.kruskal(*groups.values())
    return {"n_algorithms": len(groups), "runs_per_algorithm": int(min(len(g) for g in groups.values())),
            "statistic": float(statistic), "p_value": float(p_value)}


def compare_with_baseline(groups, baseline=RANDOM, alpha=0.05):
    if baseline not in groups:
        raise ValueError(f"Baseline {baseline!r} has no runs")
    base = groups[baseline]
    rows = []
    for name, scores in groups.items():
        if name == baseline:
            continue
        p_value = stats.mannwhitneyu(scores, base, alternative="two-sided").pvalue
        rows.append({"algorithm": name, "mean": float(np.mean(scores)), "baseline_mean": float(np.mean(base)),
                     "difference": float(np.mean(scores) - np.mean(base)), "p_value": float(p_value)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table.p_value)
    table["significant"] = table.p_holm < alpha
    return table.sort_values("difference", ascending=False, kind="stable").reset_index(drop=True)


def pcap_runs(space, path=None):
    if space not in PCAP_SEARCH_PATHS:
        raise ValueError(f"Unknown search space {space!r}; choose from {list(PCAP_SEARCH_PATHS)}")
    return pd.read_csv(PCAP_SEARCH_PATHS[space] if path is None else path)


def pcap_groups(runs, budget, score_column="best_f1"):
    chosen = runs[runs.budget == budget]
    if chosen.empty:
        raise ValueError(f"No runs with budget {budget}")
    return {name: part[score_column].to_numpy() for name, part in chosen.groupby("algorithm")}


def pcap_kruskal(space, path=None, score_column="best_f1"):
    runs = pcap_runs(space, path)
    rows = []
    for budget in sorted(runs.budget.unique()):
        groups = pcap_groups(runs, budget, score_column)
        best = max(groups, key=lambda name: groups[name].mean())
        rows.append({"space": space, "budget": int(budget), **kruskal_test(groups),
                     "best_algorithm": best, "best_mean": float(groups[best].mean())})
    return pd.DataFrame(rows)


def pcap_against_random(space, budget, path=None, score_column="best_f1", alpha=0.05):
    groups = pcap_groups(pcap_runs(space, path), budget, score_column)
    return compare_with_baseline(groups, RANDOM, alpha)


def landsat_rank_table(tasks=LANDSAT_TASKS, path=LANDSAT_MH_PATH, score_column="macro_f1", alpha=0.05):
    rows = []
    for task in tasks:
        table = landsat_table(task, path, score_column)
        cd = nemenyi_critical_difference(table.shape[1], table.shape[0], alpha)
        for algorithm, rank in average_ranks(table).items():
            rows.append({"task": task, "algorithm": algorithm, "average_rank": float(rank),
                         "mean_score": float(table[algorithm].mean()), "critical_difference": cd})
    return pd.DataFrame(rows)


def run_statistical_tests(save=True, out_dir=STATS_DIR, alpha=0.05, show=print):
    tables = {"landsat_friedman": landsat_friedman(),
              "landsat_ranks": landsat_rank_table(alpha=alpha),
              "landsat_nemenyi": pd.concat([landsat_nemenyi(task, alpha=alpha).assign(task=task)
                                            for task in LANDSAT_TASKS], ignore_index=True),
              "pcap_kruskal": pd.concat([pcap_kruskal(space) for space in PCAP_SEARCH_PATHS], ignore_index=True)}
    comparisons = []
    for space in PCAP_SEARCH_PATHS:
        for budget in sorted(pcap_runs(space).budget.unique()):
            comparisons.append(pcap_against_random(space, budget, alpha=alpha).assign(space=space, budget=int(budget)))
    tables["pcap_vs_random"] = pd.concat(comparisons, ignore_index=True)
    for name, table in tables.items():
        show(f"{name}: {len(table)} rows")
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
        show(f"Saved {len(tables)} tables to {out_dir}")
    return tables


def cd_groups(ranks, cd):
    values = np.sort(np.asarray(ranks, dtype=float))
    groups = []
    last_end = 0
    for start in range(len(values)):
        end = start
        while end + 1 < len(values) and values[end + 1] - values[start] <= cd:
            end += 1
        if end > start and end > last_end:
            groups.append((start, end))
            last_end = end
    return groups


def plot_critical_difference(ranks, cd, title=None):
    ranks = pd.Series(ranks, dtype=float).sort_values()
    names, values = list(ranks.index), ranks.to_numpy()
    k = len(values)
    half = (k + 1) // 2
    groups = cd_groups(values, cd)
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.32 * half))
    ax.set_xlim(0, k + 1)
    ax.set_ylim(-(half + 1.2), 2.2)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", linewidth=1)
    for tick in range(1, k + 1):
        ax.vlines(tick, 0, 0.18, color="black", linewidth=1)
        ax.text(tick, 0.3, str(tick), ha="center", va="bottom", fontsize=9)
    ax.hlines(1.3, 1, 1 + cd, color="black", linewidth=1.5)
    ax.vlines([1, 1 + cd], 1.2, 1.4, color="black", linewidth=1.5)
    ax.text(1 + cd / 2, 1.5, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    for i, (name, value) in enumerate(zip(names, values)):
        left = i < half
        row = -(i + 1) if left else -(k - i)
        edge = 0.6 if left else k + 0.4
        ax.plot([value, value, edge], [0, row, row], color="black", linewidth=0.8)
        ax.text(edge - 0.08 if left else edge + 0.08, row, f"{name} ({value:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)
    for level, (start, end) in enumerate(groups):
        y = -0.35 - 0.2 * level
        ax.hlines(y, values[start] - 0.05, values[end] + 0.05, color="#c0392b", linewidth=3)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def save_cd_diagrams(ranks_path=STATS_DIR / "landsat_ranks.csv", out_dir=STATS_DIR, show=print):
    table = pd.read_csv(ranks_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    titles = {"fs": "Feature selection", "threshold": "Tuning the threshold t",
              "components": "Tuning the components k", "joint": "Tuning t and k together"}
    paths = []
    for task, part in table.groupby("task", sort=False):
        ranks = part.set_index("algorithm").average_rank
        fig = plot_critical_difference(ranks, float(part.critical_difference.iloc[0]),
                                       f"Landsat: {titles.get(task, task)} (average rank, 1 = best)")
        path = out_dir / f"cd_landsat_{task}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        show(f"Saved {path}")
    return paths
