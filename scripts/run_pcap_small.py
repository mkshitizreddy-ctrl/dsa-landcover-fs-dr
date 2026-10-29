"""Pilot experiment on pcap.csv: 5 feature columns (3rd to 7th) x {RandomForest, XGBoost}.

With --pca N, also compresses ALL usable features into N PCA components (fit on
training rows only), so feature selection (5 original columns) can be compared
with dimensionality reduction (5 components) at the same input size.

Design: take 5 original feature columns starting from the 3rd column
(counting from 1) and train XGBoost and Random Forest. By default the
same models are also trained on all usable features, as a reference point.

Pipeline (course reference pipeline): median imputation -> StandardScaler
-> model. Dataset fixes:
  * Agent (1 hash) and Tsunami (3 hashes) are dropped: a grouped split cannot
    place them in both training and test.
  * Splits are grouped by Hash, so rows from one malware sample never appear in
    both training and validation/test (StratifiedGroupKFold, as in the reference pipeline).
  * 4 CV folds, not 5: Gafgyt has only 4 hashes left after the test holdout.
  * For speed, CV uses at most --max-rows-per-hash rows from each hash. The
    final models are refit on ALL development rows and scored on ALL test rows.
  * macro-F1 is the main metric (Mirai is ~69% of rows); accuracy and balanced
    accuracy are also reported.

Each (feature set, model) pair is scored once on the untouched test set.

Run from the repository root (Windows cmd):
    python scripts\\run_pcap_small.py
Options: --start-column 3  --n-columns 5  --models RF,XGB  --folds 4
         --max-rows-per-hash 30  --no-baseline  --pca 5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
NON_FEATURES = ["Hash", "Arch", "MalwareFamily"]
DROP_CLASSES = ["Agent", "Tsunami"]
METRICS = ["f1_macro", "balanced_accuracy", "accuracy"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "pcap.csv"))
    p.add_argument("--start-column", type=int, default=3, help="1-based position of the first feature column")
    p.add_argument("--n-columns", type=int, default=5)
    p.add_argument("--models", default="RF,XGB")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--max-rows-per-hash", type=int, default=30, help="0 = use every row in CV")
    p.add_argument("--no-baseline", action="store_true", help="skip the all-features reference runs")
    p.add_argument("--pca", type=int, default=0, help="also run PCA with this many components on all features (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results" / "pcap_small_v2"))
    return p.parse_args()


def make_model(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                             eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model {name!r}; use RF or XGB")


def make_pipeline(model_name, seed, n_components=0):
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    if n_components:
        # Fit inside the pipeline, so PCA only ever sees the training rows of each fit.
        steps.append(("pca", PCA(n_components=n_components, svd_solver="full")))
    steps.append(("model", make_model(model_name, seed)))
    return Pipeline(steps)


def scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def check_classes(y_part, n_classes, what):
    missing = set(range(n_classes)) - set(np.unique(y_part))
    if missing:
        raise ValueError(f"{what} is missing class ids {sorted(missing)}; try fewer --folds")


def main():
    args = parse_args()
    model_names = [m.strip() for m in args.models.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    # ---------------- load ----------------
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    missing_cols = set(NON_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df["MalwareFamily"] = df["MalwareFamily"].astype(str).str.strip()
    df["Hash"] = df["Hash"].astype(str).str.strip()
    dropped_rows = int(df["MalwareFamily"].isin(DROP_CLASSES).sum())
    df = df[~df["MalwareFamily"].isin(DROP_CLASSES)].reset_index(drop=True)

    X_all = (df.drop(columns=NON_FEATURES)
               .apply(pd.to_numeric, errors="raise")
               .replace([np.inf, -np.inf], np.nan))            # Shape [N, 39]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["MalwareFamily"])             # Shape [N]
    groups = df["Hash"].to_numpy()                             # Shape [N]
    n_classes = len(encoder.classes_)
    print(f"Rows {len(y):,} (dropped {dropped_rows} rows of {DROP_CLASSES}) | classes {n_classes}")

    # Feature columns, counted from 1 in the CSV's column order.
    start = args.start_column - 1
    selected = list(X_all.columns[start:start + args.n_columns])
    if len(selected) != args.n_columns:
        raise ValueError(f"Only {len(selected)} columns available from position {args.start_column}")
    print(f"Selected columns {args.start_column}-{args.start_column + args.n_columns - 1}: {selected}")

    # ---------------- untouched grouped test holdout (~20%) ----------------
    holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    dev_idx, test_idx = next(holdout.split(X_all, y, groups))
    check_classes(y[dev_idx], n_classes, "Development set")
    check_classes(y[test_idx], n_classes, "Test set")
    assert not set(groups[dev_idx]) & set(groups[test_idx]), "Hash leaked between dev and test"

    # Constant columns (decided on development rows only) are left out of the reference set.
    X_dev_frame = X_all.iloc[dev_idx]
    constant = [c for c in X_dev_frame.columns if X_dev_frame[c].nunique(dropna=True) <= 1]
    all_features = [c for c in X_all.columns if c not in constant]
    const_selected = [c for c in selected if c in constant]
    if const_selected:
        print(f"Warning: selected columns constant on development rows: {const_selected}")
    # name -> (columns, number of PCA components; 0 = no PCA)
    feature_sets = {f"cols{args.start_column}-{args.start_column + args.n_columns - 1}": (selected, 0)}
    if not args.no_baseline:
        feature_sets[f"all_{len(all_features)}"] = (all_features, 0)
    if args.pca:
        if not 1 <= args.pca <= len(all_features):
            raise ValueError(f"--pca must be between 1 and {len(all_features)}")
        feature_sets[f"pca{args.pca}_of_{len(all_features)}"] = (all_features, args.pca)
    print(f"Development rows {len(dev_idx):,} | test rows {len(test_idx):,} ({len(test_idx) / len(y):.1%})")

    # ---------------- per-hash cap for CV speed ----------------
    rng = np.random.default_rng(args.seed)
    if args.max_rows_per_hash > 0:
        order = rng.permutation(dev_idx)
        rank = pd.Series(groups[order]).groupby(groups[order]).cumcount().to_numpy()
        cv_idx = np.sort(order[rank < args.max_rows_per_hash])
    else:
        cv_idx = dev_idx
    y_cv, g_cv = y[cv_idx], groups[cv_idx]
    print(f"CV rows {len(cv_idx):,} (max {args.max_rows_per_hash or 'all'} rows per hash)")
    print(pd.Series(encoder.inverse_transform(y_cv)).value_counts().to_string())

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(cv.split(cv_idx, y_cv, g_cv))
    for i, (tr, va) in enumerate(splits, start=1):
        check_classes(y_cv[tr], n_classes, f"Fold {i} training")
        check_classes(y_cv[va], n_classes, f"Fold {i} validation")
        assert not set(g_cv[tr]) & set(g_cv[va]), f"Hash leaked in fold {i}"

    # ---------------- CV and test for each feature set x model ----------------
    cv_rows, test_rows = [], []
    for set_name, (cols, n_comp) in feature_sets.items():
        X_cv = X_all.iloc[cv_idx][cols].to_numpy(dtype=float)       # Shape [N_cv, len(cols)]
        X_dev = X_all.iloc[dev_idx][cols].to_numpy(dtype=float)     # Shape [N_dev, len(cols)]
        X_test = X_all.iloc[test_idx][cols].to_numpy(dtype=float)   # Shape [N_test, len(cols)]
        for name in model_names:
            t0 = time.perf_counter()
            fold_scores = []
            for tr, va in splits:
                pipe = make_pipeline(name, args.seed, n_comp).fit(X_cv[tr], y_cv[tr])
                fold_scores.append(scores(y_cv[va], pipe.predict(X_cv[va])))
            fs = pd.DataFrame(fold_scores)
            n_inputs = n_comp or len(cols)
            row = {"features": set_name, "n_features": n_inputs, "model": name}
            for metric in METRICS:
                row[f"cv_{metric}_mean"] = fs[metric].mean()
                row[f"cv_{metric}_sd"] = fs[metric].std(ddof=1)
            for i, v in enumerate(fs["f1_macro"], start=1):
                row[f"fold_{i}_f1_macro"] = v
            row["cv_seconds"] = round(time.perf_counter() - t0, 1)
            cv_rows.append(row)
            pd.DataFrame(cv_rows).to_csv(out / "cv_results.csv", index=False)
            print(f"\nCV  {set_name:9} {name:3}  macro-F1 {row['cv_f1_macro_mean']:.4f} ± "
                  f"{row['cv_f1_macro_sd']:.4f}  bal-acc {row['cv_balanced_accuracy_mean']:.4f}  "
                  f"acc {row['cv_accuracy_mean']:.4f}  ({row['cv_seconds']}s)", flush=True)

            # Refit on all development rows, score the test set once.
            t0 = time.perf_counter()
            pipe = make_pipeline(name, args.seed, n_comp).fit(X_dev, y[dev_idx])
            pred = pipe.predict(X_test)
            s = scores(y[test_idx], pred)
            explained = (float(pipe.named_steps["pca"].explained_variance_ratio_.sum())
                         if n_comp else None)
            test_rows.append({"features": set_name, "n_features": n_inputs, "model": name,
                              **{f"test_{m}": s[m] for m in METRICS},
                              "pca_explained_variance": explained,
                              "fit_seconds": round(time.perf_counter() - t0, 1)})
            pd.DataFrame(test_rows).to_csv(out / "test_results.csv", index=False)
            report = classification_report(y[test_idx], pred, labels=np.arange(n_classes),
                                           target_names=encoder.classes_, zero_division=0)
            (out / f"test_report_{set_name}_{name}.txt").write_text(report, encoding="utf-8")
            pd.DataFrame(confusion_matrix(y[test_idx], pred, labels=np.arange(n_classes)),
                         index=encoder.classes_, columns=encoder.classes_
                         ).to_csv(out / f"test_confusion_{set_name}_{name}.csv")
            if explained is not None:
                print(f"PCA keeps {explained:.1%} of the variance of {len(cols)} scaled features")
            print(f"TEST {set_name:9} {name:3}  macro-F1 {s['f1_macro']:.4f}  bal-acc "
                  f"{s['balanced_accuracy']:.4f}  acc {s['accuracy']:.4f}  ({test_rows[-1]['fit_seconds']}s)")
            print(report, flush=True)

    # ---------------- summary ----------------
    summary = pd.DataFrame(cv_rows)[["features", "n_features", "model", "cv_f1_macro_mean", "cv_f1_macro_sd",
                                     "cv_balanced_accuracy_mean", "cv_accuracy_mean"]].merge(
        pd.DataFrame(test_rows)[["features", "model", "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]],
        on=["features", "model"])
    summary.to_csv(out / "summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved to {out} | total {time.perf_counter() - started_all:.0f}s")


if __name__ == "__main__":
    main()
