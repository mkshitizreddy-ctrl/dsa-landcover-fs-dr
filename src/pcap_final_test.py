"""Final evaluation of the selected configurations on the held-out test set.

From the grid (results/pcap_grid_v1/grid_summary.csv) it takes:
  * the best overall configuration by CV accuracy (the reference pipeline's rule),
  * the best configuration of each classifier, labelled C1 (RF), C2 (XGB),
    C3 (DT) and C4 (SVM), by CV accuracy,
  * the same choices by CV macro-F1 (better for these imbalanced classes).
Each distinct configuration is refitted on the development rows and scored ONCE
on the test rows from data/pcap_splits.npz, which no earlier step has used.

As in the reference pipeline, models are refitted on ALL development rows, with two
unavoidable exceptions:
  * SVM training grows roughly with n^2, so SVM trains on a stratified
    subsample of --svm-max-rows development rows.
  * KernelPCA needs an n x n kernel matrix, so it is fitted on --kpca-max-train
    rows (as in the grid); every training and test row is then projected.

Outputs in results/pcap_final_v1/: test_results.csv, one classification report
and one confusion matrix per configuration, best_configuration.csv, and
best_model.joblib for the selected model (best overall by CV accuracy).

Run from the repository root:
    python src\\pcap_final_test.py
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcap_data import load_pcap, load_splits  # noqa: E402
from pcap_grid import CLASSIFIERS, fit_reducer, make_classifier, make_scaler, transform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = ["scaler", "reduction", "k", "classifier"]
CLASSIFIER_TAG = {"RF": "C1", "XGB": "C2", "DT": "C3", "SVM": "C4"}


def select_configs(grid):
    """Configurations to test, with every reason each one was chosen."""
    picks = {}
    for metric, label in (("accuracy_mean", "accuracy"), ("f1_macro_mean", "macro-F1")):
        ranked = grid.sort_values(metric, ascending=False, kind="stable")
        top = ranked.iloc[0]
        picks.setdefault(tuple(top[CFG]), []).append(f"best overall by {label}")
        for clf in CLASSIFIERS:
            row = ranked[ranked.classifier == clf].iloc[0]
            picks.setdefault(tuple(row[CFG]), []).append(f"{CLASSIFIER_TAG[clf]} ({clf}) by {label}")
    out = []
    for cfg, reasons in picks.items():
        row = grid[(grid.scaler == cfg[0]) & (grid.reduction == cfg[1]) &
                   (grid.k == cfg[2]) & (grid.classifier == cfg[3])].iloc[0]
        out.append({"scaler": cfg[0], "reduction": cfg[1], "k": int(cfg[2]), "classifier": cfg[3],
                    "chosen_as": "; ".join(reasons), "cv_accuracy": row.accuracy_mean,
                    "cv_f1_macro": row.f1_macro_mean, "cv_balanced_accuracy": row.balanced_accuracy_mean})
    return out


def fit_config(cfg, X_train, y_train, seed=42, kpca_max_train=8000, svm_max_rows=60000):
    """Fit imputer -> scaler -> reduction -> classifier. Returns (pipeline, n_train_rows)."""
    if cfg["classifier"] == "SVM" and svm_max_rows and len(y_train) > svm_max_rows:
        rows, _ = train_test_split(np.arange(len(y_train)), train_size=svm_max_rows,
                                   stratify=y_train, random_state=seed)
        X_train, y_train = X_train[rows], y_train[rows]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train)
    A = imputer.transform(X_train)
    scaler = make_scaler(cfg["scaler"]).fit(A)
    B = scaler.transform(A)                                            # Shape [N_train, D]
    reducer = fit_reducer(cfg["reduction"], B, y_train, cfg["k"], seed=seed,
                          kpca_max_train=kpca_max_train)
    Z = transform(reducer, B)                                          # Shape [N_train, k]
    model = make_classifier(cfg["classifier"], seed).fit(Z, y_train)
    pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("reducer", reducer), ("model", model)])
    return pipe, len(y_train)


def predict_chunked(pipe, X, chunk=4000):
    """Chunked prediction keeps the KernelPCA cross-kernel small in memory."""
    return np.concatenate([pipe.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "pcap.csv"))
    p.add_argument("--splits", default=str(ROOT / "data" / "pcap_splits.npz"))
    p.add_argument("--grid", default=str(ROOT / "results" / "pcap_grid_v1" / "grid_summary.csv"))
    p.add_argument("--svm-max-rows", type=int, default=60000)
    p.add_argument("--kpca-max-train", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results" / "pcap_final_v1"))
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if (out / "test_results.csv").exists():
        raise SystemExit(f"{out / 'test_results.csv'} exists: the test set has already been used. "
                         "Use a new --out only if you mean to evaluate again.")
    out.mkdir(parents=True, exist_ok=True)

    data = load_pcap(args.data)
    splits = load_splits(args.splits, n_rows=len(data.y))
    X = data.X[splits.features].to_numpy(dtype=float)                 # Shape [N, D]
    y, names = data.y, data.class_names
    X_dev, y_dev = X[splits.dev_idx], y[splits.dev_idx]
    X_test, y_test = X[splits.test_idx], y[splits.test_idx]
    print(f"Development rows {len(y_dev):,} | test rows {len(y_test):,} | features {X.shape[1]}")

    grid = pd.read_csv(args.grid)
    if len(grid) != 924:
        raise ValueError(f"Grid summary has {len(grid)} configurations, expected 924")
    configs = select_configs(grid)
    print(f"Configurations to evaluate: {len(configs)}")
    for c in configs:
        print(f"  {c['scaler']:10} {c['reduction']:9} k={c['k']:<2} {c['classifier']:3}  <- {c['chosen_as']}")

    rows, fitted = [], {}
    for i, cfg in enumerate(configs, start=1):
        t0 = time.perf_counter()
        pipe, n_train = fit_config(cfg, X_dev, y_dev, seed=args.seed,
                                   kpca_max_train=args.kpca_max_train, svm_max_rows=args.svm_max_rows)
        fit_seconds = time.perf_counter() - t0
        pred = predict_chunked(pipe, X_test)
        tag = f"{cfg['scaler']}_{cfg['reduction']}_k{cfg['k']}_{cfg['classifier']}"
        report = classification_report(y_test, pred, labels=np.arange(len(names)), target_names=names,
                                       zero_division=0)
        (out / f"test_report_{tag}.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(confusion_matrix(y_test, pred, labels=np.arange(len(names))),
                     index=names, columns=names).to_csv(out / f"test_confusion_{tag}.csv")
        row = {**cfg, "train_rows": n_train,
               "test_accuracy": accuracy_score(y_test, pred),
               "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
               "test_f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
               "fit_seconds": round(fit_seconds, 1),
               "total_seconds": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        fitted[tag] = pipe
        pd.DataFrame(rows).to_csv(out / "test_results.csv", index=False)
        print(f"\n[{i}/{len(configs)}] {tag}  (trained on {n_train:,} rows, {row['total_seconds']}s)")
        print(f"  CV   accuracy {cfg['cv_accuracy']:.4f}  macro-F1 {cfg['cv_f1_macro']:.4f}")
        print(f"  TEST accuracy {row['test_accuracy']:.4f}  macro-F1 {row['test_f1_macro']:.4f}  "
              f"balanced {row['test_balanced_accuracy']:.4f}")
        print(report, flush=True)

    # Selected model: best overall by CV accuracy.
    selected = next(r for r in rows if "best overall by accuracy" in r["chosen_as"])
    tag = f"{selected['scaler']}_{selected['reduction']}_k{selected['k']}_{selected['classifier']}"
    pd.DataFrame([selected]).to_csv(out / "best_configuration.csv", index=False)
    joblib.dump({"pipeline": fitted[tag], "class_names": names, "feature_names": splits.features,
                 "configuration": {c: selected[c] for c in CFG}}, out / "best_model.joblib", compress=3)

    table = pd.DataFrame(rows)[["classifier", "scaler", "reduction", "k", "chosen_as", "cv_accuracy",
                                "test_accuracy", "cv_f1_macro", "test_f1_macro", "test_balanced_accuracy"]]
    print("\nFINAL TEST RESULTS")
    print(table.round(4).to_string(index=False))
    print(f"\nSelected model (highest CV accuracy): {tag}. Saved as best_model.joblib")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
