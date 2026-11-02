"""Write all results of the pcap study into one readable text file.

It only reads the saved result files; no model is trained again.

Run from the repository root:
    python scripts\\write_pcap_report.py

Output:
    results\\pcap_results.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "pcap_results.txt"

# Readable names for the columns that appear in the tables.
NAMES = {
    "scaler": "Scaler", "reduction": "Reduction", "k": "k", "classifier": "Classifier",
    "f1_macro_mean": "CV macro-F1", "f1_macro_sd": "SD",
    "balanced_accuracy_mean": "CV bal. acc.", "accuracy_mean": "CV accuracy",
}

report = []  # the text file, one line at a time


def heading(title):
    report.extend(["", "=" * 80, title, "=" * 80])


def table(df):
    report.append(df.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# 1. Data and protocol
# ----------------------------------------------------------------------------
splits = np.load(ROOT / "data" / "pcap_splits.npz")
meta = json.loads(str(splits["meta"]))

heading("1. DATA AND PROTOCOL")
report += [
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

# ----------------------------------------------------------------------------
# 2. Pilot experiment: 5 columns vs all features vs 5 PCA components
# ----------------------------------------------------------------------------
pilot = pd.read_csv(RESULTS / "pcap_small_v2" / "summary.csv")
pilot = pilot[["features", "n_features", "model", "test_f1_macro",
               "test_balanced_accuracy", "test_accuracy"]]
pilot.columns = ["Inputs", "Count", "Model", "Test macro-F1", "Test bal. acc.", "Test accuracy"]

heading("2. PILOT EXPERIMENT (Random Forest and XGBoost, test set)")
report.append("cols3-7 = 5 original columns (3rd to 7th); all_36 = all features;"
              " pca5_of_36 = 5 PCA components")
report.append("")
table(pilot)

# ----------------------------------------------------------------------------
# 3. Exhaustive grid: scaler x reduction x k x classifier
# ----------------------------------------------------------------------------
grid = pd.read_csv(RESULTS / "pcap_grid_v1" / "grid_summary.csv")
columns = list(NAMES)
# kind="stable" breaks exact ties the same way as pcap_final_test.py does.
by_f1 = grid.sort_values("f1_macro_mean", ascending=False, kind="stable")
by_accuracy = grid.sort_values("accuracy_mean", ascending=False, kind="stable")

heading(f"3. GRID SEARCH ({len(grid)} configurations, each scored by 4-fold CV)")
report.append("Scalers: Standard, MinMax, Normalizer | Reductions: PCA, KernelPCA, LDA"
              " | Classifiers: RF, XGB, DT, SVM")

report += ["", "Top 10 configurations by CV macro-F1:"]
table(by_f1.head(10)[columns].rename(columns=NAMES))

report += ["", "Best configuration of each classifier, by CV accuracy (C1 = RF, C2 = XGB,"
           " C3 = DT, C4 = SVM):"]
table(by_accuracy.groupby("classifier", sort=False).head(1)[columns].rename(columns=NAMES))

report += ["", "Best configuration of each classifier, by CV macro-F1:"]
table(by_f1.groupby("classifier", sort=False).head(1)[columns].rename(columns=NAMES))

report += ["", "Best configuration of each reduction method, by CV macro-F1:"]
table(by_f1.groupby("reduction", sort=False).head(1)[columns].rename(columns=NAMES))

report += ["", "Best configuration of each scaler, by CV macro-F1:"]
table(by_f1.groupby("scaler", sort=False).head(1)[columns].rename(columns=NAMES))

# ----------------------------------------------------------------------------
# 4. Metaheuristic search over the grid
# ----------------------------------------------------------------------------
runs = pd.read_csv(RESULTS / "pcap_mh_v1" / "runs.csv")
runs["algorithm"] = runs["algorithm"].replace({"RANDOM": "Random search"})
search = runs.groupby(["budget", "algorithm"]).agg(
    found_best=("hit_best", "mean"),
    within_001=("hit_near", "mean"),
    mean_rank=("best_rank", "mean"),
    configs_tried=("unique_configs", "mean"),
).reset_index()
search["found_best"] *= 100
search["within_001"] *= 100

best = by_f1.iloc[0]
heading("4. METAHEURISTIC SEARCH OVER THE GRID (20 seeds per algorithm)")
report += [
    f"Best grid configuration: {best.scaler} + {best.reduction}, k = {best.k}, {best.classifier}"
    f" (CV macro-F1 {best.f1_macro_mean:.3f})",
    "Each candidate's score is looked up from the grid. Cost = configurations tried"
    f" (out of {len(grid)}).",
    "found_best and within_001 are % of runs; mean_rank is the true rank of the result (1 = best).",
]
for budget, part in search.groupby("budget"):
    report += ["", f"Budget: {budget} fitness evaluations"]
    part = part.drop(columns="budget").sort_values(["found_best", "within_001"], ascending=False)
    table(part.rename(columns={"algorithm": "Algorithm", "found_best": "Found best %",
                               "within_001": "Within 0.01 %", "mean_rank": "Mean rank",
                               "configs_tried": "Configs tried"}))

# ----------------------------------------------------------------------------
# 5. Final evaluation on the held-out test set
# ----------------------------------------------------------------------------
final = pd.read_csv(RESULTS / "pcap_final_v1" / "test_results.csv")
final = final.sort_values("test_f1_macro", ascending=False)
selected = pd.read_csv(RESULTS / "pcap_final_v1" / "best_configuration.csv").iloc[0]

# Accuracy of always predicting Mirai, from a confusion matrix (row sums = test rows per family).
confusion = pd.read_csv(RESULTS / "pcap_final_v1" / "test_confusion_MinMax_PCA_k36_XGB.csv", index_col=0)
mirai_only = confusion.loc["Mirai"].sum() / confusion.values.sum()

heading("5. FINAL TEST (refitted on development rows, evaluated once)")
table(final[["classifier", "scaler", "reduction", "k", "chosen_as", "cv_accuracy",
             "cv_f1_macro", "test_accuracy", "test_balanced_accuracy", "test_f1_macro",
             "train_rows"]].rename(columns={
    "classifier": "Classifier", "scaler": "Scaler", "reduction": "Reduction",
    "chosen_as": "Selected as", "cv_accuracy": "CV acc.", "cv_f1_macro": "CV macro-F1",
    "test_accuracy": "Test acc.", "test_balanced_accuracy": "Test bal. acc.",
    "test_f1_macro": "Test macro-F1", "train_rows": "Train rows"}))
report += [
    "",
    f"Selected model (highest CV accuracy): {selected.classifier} on {selected.scaler} +"
    f" {selected.reduction}, k = {selected.k}  ->  test accuracy {selected.test_accuracy:.3f},"
    f" macro-F1 {selected.test_f1_macro:.3f}",
    f"Reference: always predicting Mirai gives accuracy {mirai_only:.3f}.",
    "Note: SVM trained on a 60,000-row subsample; KernelPCA fitted on 8,000 rows.",
]

report += ["", "Per-family results on the test set:"]
for row in final.itertuples():
    name = f"{row.scaler}_{row.reduction}_k{row.k}_{row.classifier}"
    report += ["", f"--- {row.classifier}: {row.scaler} + {row.reduction}, k = {row.k} ---"]
    report.append((RESULTS / "pcap_final_v1" / f"test_report_{name}.txt").read_text(encoding="utf-8").rstrip())

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
title = ["PCAP MALWARE FAMILY CLASSIFICATION: RESULTS", "DSA Project, M.Tech AIML, Bennett University"]
OUTPUT.write_text("\n".join(title + report) + "\n", encoding="utf-8")
print(f"Saved {OUTPUT}")
