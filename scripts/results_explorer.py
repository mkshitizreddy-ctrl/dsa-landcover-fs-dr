from pathlib import Path

import numpy as np
import pandas as pd

from pcap_data import SPLITS_PATH, load_splits

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRID_SUMMARY_PATH = RESULTS / "pcap_grid_v1" / "grid_summary.csv"
TEST_COMPARISON_PATH = RESULTS / "pcap_fs_final_v1" / "comparison_with_grid_test.csv"
WEIGHTING_PATH = RESULTS / "pcap_weighting_v1" / "summary.csv"
CONFUSION_DIRS = (RESULTS / "pcap_final_v1", RESULTS / "pcap_fs_final_v1")
BEST_CONFUSION_PATH = RESULTS / "pcap_final_v1" / "test_confusion_Normalizer_PCA_k14_DT.csv"
STATS_DIR = RESULTS / "stats_v1"
SELECTION_METHODS = ("MI_top", "HeapBFS", "Mask")
GRID_COLUMNS = ["scaler", "reduction", "k", "classifier", "f1_macro_mean", "f1_macro_sd",
                "balanced_accuracy_mean", "accuracy_mean"]


def split_summary(splits_path=SPLITS_PATH):
    splits = load_splits(splits_path)
    meta = splits.meta
    rows = [("Packets after cleaning", meta["n_rows"]),
            ("Removed (Agent and Tsunami)", meta["dropped_rows"]),
            ("Development", len(splits.dev_idx)),
            ("Test, used once", len(splits.test_idx)),
            ("Cross-validation sample, at most 5 per hash", len(splits.sample_idx))]
    rows += [(f"Fold {i} validation", len(valid)) for i, (_, valid) in enumerate(splits.folds, start=1)]
    return {"table": pd.DataFrame(rows, columns=["part", "packets"]), "families": list(meta["class_names"]),
            "features": list(splits.features), "constant_columns": list(meta["constant_columns_dropped"])}


def read_confusion(path):
    return pd.read_csv(path, index_col=0)


def class_counts(confusion):
    return confusion.sum(axis=1).rename("packets")


def majority_accuracy(confusion):
    counts = class_counts(confusion)
    return float(counts.max() / counts.sum())


def row_percentages(confusion):
    totals = confusion.sum(axis=1).replace(0, np.nan)
    return confusion.div(totals, axis=0).mul(100).fillna(0.0)


def confusion_files(folders=CONFUSION_DIRS):
    files = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("test_confusion_*.csv")):
            files[path.stem.removeprefix("test_confusion_").replace("_", " ")] = path
    return files


def load_grid(path=GRID_SUMMARY_PATH):
    return pd.read_csv(path)


def filter_grid(grid, scalers=None, reductions=None, classifiers=None):
    keep = pd.Series(True, index=grid.index)
    for column, values in (("scaler", scalers), ("reduction", reductions), ("classifier", classifiers)):
        if values:
            keep &= grid[column].isin(values)
    return grid[keep]


def top_configs(grid, n=10, metric="f1_macro_mean"):
    ordered = grid.sort_values(metric, ascending=False, kind="stable")
    return ordered[GRID_COLUMNS].head(n).reset_index(drop=True)


def best_by_k(grid, metric="f1_macro_mean"):
    return grid.groupby(["reduction", "k"], as_index=False)[metric].max()


def describe_setting(method, param):
    if method == "MI_top":
        return f"MI top-k, k = {int(float(param))}"
    if method == "HeapBFS":
        return f"heap + BFS, t = {float(param):.2f}"
    if method == "Mask":
        return f"metaheuristic mask, {param}"
    return f"{method}, k = {int(float(param))}"


def final_test_comparison(path=TEST_COMPARISON_PATH):
    table = pd.read_csv(path)
    table.insert(0, "family", np.where(table.method.isin(SELECTION_METHODS), "selection", "extraction"))
    table.insert(2, "setting", [describe_setting(m, p) for m, p in zip(table.method, table.param)])
    columns = ["family", "scaler", "setting", "classifier", "n_features", "chosen_as", "cv_f1_macro",
               "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]
    return table[columns].sort_values("test_f1_macro", ascending=False, kind="stable").reset_index(drop=True)


def weighting_table(path=WEIGHTING_PATH, metric="f1_macro"):
    table = pd.read_csv(path)
    table["configuration"] = [f"{s} + {describe_setting(m, p)} + {c}" for s, m, p, c in
                              zip(table.scaler, table.method, table.param, table.classifier)]
    wide = table.pivot_table(index=["family", "configuration"], columns="scheme", values=metric, sort=False)
    return wide[["none", "sqrt", "balanced"]].reset_index()


def stats_tables(folder=STATS_DIR):
    return {path.stem: pd.read_csv(path) for path in sorted(Path(folder).glob("*.csv"))}


def cd_figures(folder=STATS_DIR):
    return sorted(Path(folder).glob("cd_*.png"))
