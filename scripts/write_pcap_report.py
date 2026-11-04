import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_PATH = RESULTS / "pcap_results.txt"
NAMES = {
    "scaler": "Scaler", "reduction": "Reduction", "k": "k", "classifier": "Classifier",
    "f1_macro_mean": "CV macro-F1", "f1_macro_sd": "SD",
    "balanced_accuracy_mean": "CV bal. acc.", "accuracy_mean": "CV accuracy",
}


def heading(title):
    return ["", "=" * 80, title, "=" * 80]


def table(df):
    return [df.round(3).to_string(index=False)]


def data_section():
    splits = np.load(ROOT / "data" / "pcap_splits.npz")
    meta = json.loads(str(splits["meta"]))
    return heading("1. DATA AND PROTOCOL") + [
        "Dataset: pcap.csv (one row = one network packet; Hash = malware sample)",
        f"Rows used: {meta['n_rows']:,}   (removed {meta['dropped_rows']} rows of Agent and Tsunami:"
        " too few hashes for a grouped split)",
        f"Families: {', '.join(meta['class_names'])}",
        f"Features: {len(meta['features'])}   (constant columns removed: "
        f"{', '.join(meta['constant_columns_dropped'])})",
        f"Development rows: {len(splits['dev_idx']):,}   Test rows: {len(splits['test_idx']):,}"
        "   (split grouped by Hash; test set used once)",
        f"Grid sample: {len(splits['sample_idx']):,} rows (at most {meta['rows_per_hash']} per hash),"
        f" {meta['cv_folds']}-fold CV grouped by Hash",
        "Main metric: macro-F1 (every family counts equally); accuracy also reported.",
    ]


def pilot_section():
    pilot = pd.read_csv(RESULTS / "pcap_small_v2" / "summary.csv")
    pilot = pilot[["features", "n_features", "model", "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]]
    pilot.columns = ["Inputs", "Count", "Model", "Test macro-F1", "Test bal. acc.", "Test accuracy"]
    return (heading("2. PILOT EXPERIMENT (Random Forest and XGBoost, test set)")
            + ["cols3-7 = 5 original columns (3rd to 7th); all_36 = all features; pca5_of_36 = 5 PCA components", ""]
            + table(pilot))


def best_of_each(ranked, column):
    return ranked.groupby(column, sort=False).head(1)[list(NAMES)].rename(columns=NAMES)


def grid_section(grid):
    by_f1 = grid.sort_values("f1_macro_mean", ascending=False, kind="stable")
    by_accuracy = grid.sort_values("accuracy_mean", ascending=False, kind="stable")
    lines = heading(f"3. GRID SEARCH ({len(grid)} configurations, each scored by 4-fold CV)")
    lines.append("Scalers: Standard, MinMax, Normalizer | Reductions: PCA, KernelPCA, LDA | Classifiers: RF, XGB, DT, SVM")
    parts = [
        ("Top 10 configurations by CV macro-F1:", by_f1.head(10)[list(NAMES)].rename(columns=NAMES)),
        ("Best configuration of each classifier, by CV accuracy (C1 = RF, C2 = XGB, C3 = DT, C4 = SVM):",
         best_of_each(by_accuracy, "classifier")),
        ("Best configuration of each classifier, by CV macro-F1:", best_of_each(by_f1, "classifier")),
        ("Best configuration of each reduction method, by CV macro-F1:", best_of_each(by_f1, "reduction")),
        ("Best configuration of each scaler, by CV macro-F1:", best_of_each(by_f1, "scaler")),
    ]
    for title, df in parts:
        lines += ["", title] + table(df)
    return lines


def search_section(grid):
    runs = pd.read_csv(RESULTS / "pcap_mh_v1" / "runs.csv")
    runs["algorithm"] = runs["algorithm"].replace({"RANDOM": "Random search"})
    search = runs.groupby(["budget", "algorithm"]).agg(
        found_best=("hit_best", "mean"), within_001=("hit_near", "mean"),
        mean_rank=("best_rank", "mean"), configs_tried=("unique_configs", "mean")).reset_index()
    search[["found_best", "within_001"]] *= 100
    best = grid.sort_values("f1_macro_mean", ascending=False, kind="stable").iloc[0]
    lines = heading("4. METAHEURISTIC SEARCH OVER THE GRID (20 seeds per algorithm)") + [
        f"Best grid configuration: {best.scaler} + {best.reduction}, k = {best.k}, {best.classifier}"
        f" (CV macro-F1 {best.f1_macro_mean:.3f})",
        f"Each candidate's score is looked up from the grid. Cost = configurations tried (out of {len(grid)}).",
        "found_best and within_001 are % of runs; mean_rank is the true rank of the result (1 = best).",
    ]
    for budget, part in search.groupby("budget"):
        part = part.drop(columns="budget").sort_values(["found_best", "within_001"], ascending=False)
        lines += ["", f"Budget: {budget} fitness evaluations"] + table(part.rename(columns={
            "algorithm": "Algorithm", "found_best": "Found best %", "within_001": "Within 0.01 %",
            "mean_rank": "Mean rank", "configs_tried": "Configs tried"}))
    return lines


def final_section():
    final_dir = RESULTS / "pcap_final_v1"
    final = pd.read_csv(final_dir / "test_results.csv").sort_values("test_f1_macro", ascending=False)
    selected = pd.read_csv(final_dir / "best_configuration.csv").iloc[0]
    confusion = pd.read_csv(final_dir / "test_confusion_MinMax_PCA_k36_XGB.csv", index_col=0)
    mirai_only = confusion.loc["Mirai"].sum() / confusion.values.sum()
    columns = {"classifier": "Classifier", "scaler": "Scaler", "reduction": "Reduction", "k": "k",
               "chosen_as": "Selected as", "cv_accuracy": "CV acc.", "cv_f1_macro": "CV macro-F1",
               "test_accuracy": "Test acc.", "test_balanced_accuracy": "Test bal. acc.",
               "test_f1_macro": "Test macro-F1", "train_rows": "Train rows"}
    lines = heading("5. FINAL TEST (refitted on development rows, evaluated once)")
    lines += table(final[list(columns)].rename(columns=columns))
    lines += ["",
              f"Selected model (highest CV accuracy): {selected.classifier} on {selected.scaler} +"
              f" {selected.reduction}, k = {selected.k}  ->  test accuracy {selected.test_accuracy:.3f},"
              f" macro-F1 {selected.test_f1_macro:.3f}",
              f"Reference: always predicting Mirai gives accuracy {mirai_only:.3f}.",
              "Note: SVM trained on a 60,000-row subsample; KernelPCA fitted on 8,000 rows.",
              "", "Per-family results on the test set:"]
    for row in final.itertuples():
        name = f"{row.scaler}_{row.reduction}_k{row.k}_{row.classifier}"
        lines += ["", f"--- {row.classifier}: {row.scaler} + {row.reduction}, k = {row.k} ---",
                  (final_dir / f"test_report_{name}.txt").read_text(encoding="utf-8").rstrip()]
    return lines


def write_pcap_report(out_path=REPORT_PATH):
    grid = pd.read_csv(RESULTS / "pcap_grid_v1" / "grid_summary.csv")
    lines = ["PCAP MALWARE FAMILY CLASSIFICATION: RESULTS", "DSA Project, M.Tech AIML, Bennett University"]
    lines += data_section() + pilot_section() + grid_section(grid) + search_section(grid) + final_section()
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(out_path)
