from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from landsat_data import ROOT, load_landsat, standardize
from landsat_explorer import feature_names, select, threshold_sweep
from meta_fs import ALGORITHMS
from pcap_feature_selection import feature_statistics

RESULTS = ROOT / "results"
FIGURE_DIR = ROOT / "report" / "figures"
INK, MUTED, GRIDLINE, GREY = "#0b0b0b", "#52514e", "#e1e0d9", "#b9bcc2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = ["#9ec5f4", "#3987e5", "#184f95"]
ORDER = list(ALGORITHMS) + ["RANDOM"]
FAMILIES = ["Benign", "DarkNexus", "Gafgyt", "Generic", "Mirai", "Unknown"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, out_dir, name):
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_table(table, out_dir, name):
    table.to_csv(Path(out_dir) / f"{name}.csv", index=False)
    return Path(out_dir) / f"{name}.csv"


def landsat_statistics():
    X, y, _, _ = load_landsat()
    (Z,) = standardize(X)
    return feature_statistics(Z, y)


def figure_mi(scores, C, out_dir, t=0.9):
    kept = set(select(scores, C, t)["features"])
    table = pd.DataFrame({"feature": feature_names(len(scores)), "mi": scores})
    table["kept"] = [j in kept for j in range(len(scores))]
    table = table.sort_values("mi", kind="stable")
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    ax.barh(table.feature, table.mi, color=[SERIES[0] if k else GREY for k in table.kept], height=0.7)
    ax.set_xlabel("Mutual information with the class (nats)")
    ax.set_title(f"Landsat: relevance of each feature, kept by heap + BFS at t = {t} in blue")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    return save(fig, out_dir, "F3_landsat_mutual_information")


def figure_threshold_sweep(scores, C, out_dir):
    thresholds = np.round(np.arange(0.50, 1.001, 0.01), 2)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for method, label, color in (("heap_bfs", "Heap + BFS", SERIES[0]),
                                 ("per_component", "Best per component (Union-Find)", SERIES[1])):
        sweep = threshold_sweep(scores, C, thresholds, method)
        ax.plot(sweep.threshold, sweep.features_kept, color=color, linewidth=2, label=label)
    ax.axvline(0.9, color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.9, 1, " t = 0.9", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Correlation threshold t")
    ax.set_ylabel("Features kept (of 36)")
    ax.set_title("Landsat: how the threshold controls redundancy removal")
    ax.legend(loc="upper left")
    return save(fig, out_dir, "F4_landsat_threshold_sweep")


def hit_rates(table, value):
    rates = table.groupby("algorithm")[value].mean().mul(100)
    return rates.reindex([a for a in ORDER if a in rates.index])


def figure_hit_rates(out_dir):
    landsat = pd.read_csv(RESULTS / "mh_comparison_v1.csv").rename(columns={"method": "algorithm"})
    landsat["hit"] = landsat.hit_grid_optimum.astype(str).str.lower().isin(["true", "1", "1.0"])
    grid = pd.read_csv(RESULTS / "pcap_mh_v1" / "runs.csv")
    selection = pd.read_csv(RESULTS / "pcap_fs_search_v1" / "runs.csv")
    panels = [
        ("Landsat: threshold t", hit_rates(landsat[landsat.task == "threshold"], "hit")),
        ("Landsat: t and k together", hit_rates(landsat[landsat.task == "joint"], "hit")),
        ("pcap grid, budget 200", hit_rates(grid[grid.budget == 200], "hit_best")),
        ("pcap selection settings, budget 400", hit_rates(selection[selection.budget == 400], "hit_best")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), sharey=True)
    for ax, (title, rates) in zip(axes, panels):
        rates = rates.reindex(ORDER)
        colors = [GREY if a == "RANDOM" else SERIES[0] for a in rates.index]
        ax.barh(rates.index, rates.fillna(0), color=colors, height=0.7)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(rates):
            label = "n/a" if np.isnan(v) else f"{v:.0f}"
            ax.text(min(v if not np.isnan(v) else 0, 100) + 2, i, label, va="center", fontsize=7, color=MUTED)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Algorithm (grey: random search)")
    fig.supxlabel("Runs that found the exact best setting (%)", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "F7_metaheuristic_hit_rates")


def figure_convergence(out_dir, budget=400, highlight=3):
    runs = pd.read_csv(RESULTS / "pcap_mh_v1" / "runs.csv")
    runs = runs[runs.budget == budget]
    steps = [c for c in runs.columns if c.startswith("best_at_")]
    x = [int(c.rsplit("_", 1)[1]) for c in steps]
    curves = runs.groupby("algorithm")[steps].mean()
    best = curves[steps[-1]].drop("RANDOM").sort_values(ascending=False).index[:highlight]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for algorithm, values in curves.iterrows():
        if algorithm not in best and algorithm != "RANDOM":
            ax.plot(x, values, color=GREY, linewidth=0.8, alpha=0.7)
    ax.plot(x, curves.loc["RANDOM"], color=MUTED, linewidth=1.5, linestyle="--", label="Random search")
    for algorithm, color in zip(best, SERIES):
        ax.plot(x, curves.loc[algorithm], color=color, linewidth=2, marker="o", markersize=4, label=algorithm)
    best_score = pd.read_csv(RESULTS / "pcap_grid_v1" / "grid_summary.csv").f1_macro_mean.max()
    ax.axhline(best_score, color=INK, linewidth=0.8, linestyle=":")
    ax.text(x[0], best_score, " true best", va="bottom", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.set_xlabel("Distinct configurations tried (log scale); a flat line = the algorithm found no new ones")
    ax.set_ylabel("Mean best CV macro-F1 so far")
    ax.set_title(f"pcap grid: convergence over {runs.seed.nunique()} seeds (grey: other algorithms)")
    ax.legend(loc="lower right")
    return save(fig, out_dir, "F8_convergence_pcap_grid")


def figure_grid_best(out_dir):
    grid = pd.read_csv(RESULTS / "pcap_grid_v1" / "grid_summary.csv")
    best = grid.groupby(["scaler", "reduction", "classifier"]).f1_macro_mean.max().reset_index()
    scalers, reductions = ["Standard", "MinMax", "Normalizer"], ["PCA", "KernelPCA", "LDA"]
    classifiers = ["RF", "XGB", "DT", "SVM"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    offset = 0.18
    for ax, scaler in zip(axes, scalers):
        for i, (reduction, color) in enumerate(zip(reductions, SERIES)):
            part = best[(best.scaler == scaler) & (best.reduction == reduction)].set_index("classifier")
            ax.scatter(np.arange(4) + (i - 1) * offset, part.reindex(classifiers).f1_macro_mean, s=42,
                       color=color, edgecolor="white", linewidth=1, zorder=3, label=reduction)
        ax.set_xticks(np.arange(4), classifiers)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(f"{scaler} scaler", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Best CV macro-F1 over k")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.suptitle("pcap grid: best configuration of each scaler, reduction and classifier", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, "F9_pcap_grid_best")


def figure_selection_vs_extraction(out_dir):
    table = pd.read_csv(RESULTS / "pcap_fs_v1" / "comparison_with_grid.csv")
    table["family"] = np.where(table.method.isin(["MI_top", "HeapBFS"]), "selection", "extraction")
    best = table.groupby(["scaler", "classifier", "family"]).cv_f1_macro.max().unstack("family").reset_index()
    best["pair"] = best.scaler + " + " + best.classifier
    best = best.sort_values("extraction")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    y = np.arange(len(best))
    ax.hlines(y, best[["selection", "extraction"]].min(axis=1), best[["selection", "extraction"]].max(axis=1),
              color=GREY, linewidth=2)
    ax.scatter(best.extraction, y, color=SERIES[1], s=40, zorder=3, label="Best extraction (PCA, KernelPCA, LDA)")
    ax.scatter(best.selection, y, color=SERIES[0], s=40, zorder=3, label="Best selection (MI top-k, heap + BFS)")
    ax.set_yticks(y, best.pair)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Best CV macro-F1")
    wins = int((best.selection > best.extraction).sum())
    ax.set_title(f"pcap: selection is higher in {wins} of {len(best)} scaler and classifier pairs")
    ax.legend(loc="lower right", fontsize=8)
    return save(fig, out_dir, "F10_selection_vs_extraction")


def figure_weighting(out_dir):
    summary = pd.read_csv(RESULTS / "pcap_weighting_v1" / "summary.csv")
    means = summary.groupby("scheme")[[f"f1_{f}" for f in FAMILIES]].mean().reindex(["none", "sqrt", "balanced"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.27
    for i, (scheme, color) in enumerate(zip(means.index, RAMP)):
        ax.bar(np.arange(len(FAMILIES)) + (i - 1) * width, means.loc[scheme], width * 0.92, color=color,
               label=f"{scheme} weights")
    ax.set_xticks(np.arange(len(FAMILIES)), FAMILIES)
    ax.set_ylabel("Mean CV F1 over 8 configurations")
    ax.set_title("pcap: per-family F1 with no, square-root and balanced class weights")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", fontsize=8)
    return save(fig, out_dir, "F11_class_weighting")


def figure_confusion(out_dir):
    files = [("Normalizer + PCA (14) + DT (grid best)",
              RESULTS / "pcap_final_v1" / "test_confusion_Normalizer_PCA_k14_DT.csv"),
             ("Normalizer + MI top-33 + DT (our best)",
              RESULTS / "pcap_fs_final_v1" / "test_confusion_Normalizer_MI_top_k33_DT.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, (title, path) in zip(axes, files):
        counts = pd.read_csv(path, index_col=0)
        recall = counts.div(counts.sum(axis=1), axis=0)
        ax.imshow(recall.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue", ["#ffffff", "#cde2fb", "#3987e5", "#104281"]), vmin=0, vmax=1)
        ax.set_xticks(range(len(recall.columns)), recall.columns, rotation=40, ha="right")
        ax.set_yticks(range(len(recall.index)), recall.index)
        ax.grid(False)
        for i in range(len(recall)):
            for j in range(len(recall.columns)):
                v = recall.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
        ax.set_xlabel("Predicted family")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("True family")
    fig.suptitle("pcap held-out test: share of each true family's packets by prediction (rows sum to 1)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, "F12_confusion_matrices")


def growth(table, size, column, minimum):
    part = table[table[size] >= minimum]
    return float(np.polyfit(np.log(part[size]), np.log(part[column]), 1)[0])


def table_complexity(out_dir):
    by_d = pd.read_csv(RESULTS / "complexity.csv")
    landsat = pd.read_csv(RESULTS / "complexity_rows_landsat.csv")
    pcap = pd.read_csv(RESULTS / "complexity_rows_pcap.csv")
    rows = [("Mutual information", "O(n d)", None, "mi_s"),
            ("Correlation matrix", "O(n d^2)", "corr_s", "corr_s"),
            ("Correlation graph", "O(d^2)", "graph_s", "graph_s"),
            ("Heap + BFS selection", "O((d + E) log d)", "heap_bfs_select_s", "heap_bfs_select_s"),
            ("Naive selection", "O(d x kept)", "naive_s", "naive_s")]
    table = pd.DataFrame([{
        "Stage": stage, "Theory": theory,
        "Growth in d (synthetic)": round(growth(by_d, "d", d_col, 200), 2) if d_col else None,
        "Growth in n (Landsat)": round(growth(landsat, "n", n_col, landsat.n.iloc[len(landsat) // 2]), 2),
        "Growth in n (pcap)": round(growth(pcap, "n", n_col, pcap.n.iloc[len(pcap) // 2]), 2)}
        for stage, theory, d_col, n_col in rows])
    return save_table(table, out_dir, "T1_complexity")


def landsat_setting_name(setting):
    names = {"raw (all 36)": "All 36 features", "PCA only (95% var)": "PCA (95% of the variance)",
             "FS (t=0.95) + PCA": "Heap + BFS t = 0.95, then PCA", "FS (t=0.9) + PCA": "Heap + BFS t = 0.9, then PCA"}
    if setting in names:
        return names[setting]
    return setting.replace("heap+BFS t=", "Heap + BFS t = ")


def table_landsat(out_dir):
    selection = pd.read_csv(RESULTS / "fs_experiment.csv").rename(columns={"features": "dimensions"})
    pca = pd.read_csv(RESULTS / "pca_experiment.csv").rename(columns={"dims": "dimensions"})
    pca = pca[pca.setting.str.contains("PCA")]
    columns = ["setting", "dimensions", "accuracy", "macro_f1"]
    table = pd.concat([selection[columns], pca[columns]], ignore_index=True)
    table["setting"] = table.setting.map(landsat_setting_name)
    return save_table(table.round(4), out_dir, "T3_landsat_test")


def table_pcap_test(out_dir):
    table = pd.read_csv(RESULTS / "pcap_fs_final_v1" / "comparison_with_grid_test.csv")
    columns = ["scaler", "method", "param", "classifier", "n_features", "cv_f1_macro", "test_f1_macro",
               "test_balanced_accuracy", "test_accuracy"]
    return save_table(table[columns].round(4), out_dir, "T4_pcap_test")


def make_report_figures(out_dir=FIGURE_DIR, show=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()
    scores, C = landsat_statistics()
    makers = [lambda: figure_mi(scores, C, out_dir), lambda: figure_threshold_sweep(scores, C, out_dir),
              lambda: figure_hit_rates(out_dir), lambda: figure_convergence(out_dir),
              lambda: figure_grid_best(out_dir), lambda: figure_selection_vs_extraction(out_dir),
              lambda: figure_weighting(out_dir), lambda: figure_confusion(out_dir),
              lambda: table_complexity(out_dir), lambda: table_landsat(out_dir), lambda: table_pcap_test(out_dir)]
    paths = []
    for make in makers:
        paths.append(make())
        show(f"Saved {paths[-1].name}")
    return paths
