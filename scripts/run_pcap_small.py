import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pcap_data import (DATA_PATH, SPLITS_PATH, cap_rows_per_group, check_all_classes, check_no_shared_groups,
                       load_pcap, load_splits)
from pcap_grid import make_classifier

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "results" / "pcap_small_v2"
METRICS = ["f1_macro", "balanced_accuracy", "accuracy"]


def make_pipeline(model_name, seed, n_components=0):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]
    if n_components:
        steps.append(("pca", PCA(n_components=n_components, svd_solver="full")))
    steps.append(("model", make_classifier(model_name, seed)))
    return Pipeline(steps)


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def pick_columns(columns, start_column, n_columns):
    picked = list(columns[start_column - 1:start_column - 1 + n_columns])
    if len(picked) != n_columns:
        raise ValueError(f"Only {len(picked)} columns available from position {start_column}")
    return picked


def feature_sets(data, splits, start_column, n_columns, include_all, pca):
    usable = splits.features
    sets = {f"cols{start_column}-{start_column + n_columns - 1}":
            (pick_columns(data.X.columns, start_column, n_columns), 0)}
    if include_all:
        sets[f"all_{len(usable)}"] = (usable, 0)
    if pca:
        if not 1 <= pca <= len(usable):
            raise ValueError(f"pca must be between 1 and {len(usable)}")
        sets[f"pca{pca}_of_{len(usable)}"] = (usable, pca)
    return sets


def cv_folds(data, cv_idx, folds, seed):
    y_cv, g_cv = data.y[cv_idx], data.groups[cv_idx]
    n_classes = len(data.class_names)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(splitter.split(cv_idx, y_cv, g_cv))
    for i, (train, valid) in enumerate(result, start=1):
        check_all_classes(y_cv[train], n_classes, f"Fold {i} training (try fewer folds)")
        check_all_classes(y_cv[valid], n_classes, f"Fold {i} validation (try fewer folds)")
        check_no_shared_groups(g_cv, train, valid, f"fold {i}")
    return result


def cross_validate(name, X_cv, y_cv, folds, seed, n_components):
    fold_scores = [scores(y_cv[valid], make_pipeline(name, seed, n_components).fit(X_cv[train], y_cv[train])
                          .predict(X_cv[valid])) for train, valid in folds]
    table = pd.DataFrame(fold_scores)
    row = {}
    for metric in METRICS:
        row[f"cv_{metric}_mean"] = table[metric].mean()
        row[f"cv_{metric}_sd"] = table[metric].std(ddof=1)
    return row


def save_test_outputs(out_dir, set_name, name, y_test, pred, class_names):
    labels = np.arange(len(class_names))
    report = classification_report(y_test, pred, labels=labels, target_names=class_names, zero_division=0)
    (out_dir / f"test_report_{set_name}_{name}.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=class_names,
                 columns=class_names).to_csv(out_dir / f"test_confusion_{set_name}_{name}.csv")


def run_pcap_pilot(data_path=DATA_PATH, splits_path=SPLITS_PATH, start_column=3, n_columns=5, models=("RF", "XGB"),
                   folds=4, max_rows_per_hash=30, include_all=True, pca=5, seed=42, save=False, out_dir=PILOT_DIR,
                   show=print):
    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    cv_idx = cap_rows_per_group(splits.dev_idx, data.groups, max_rows_per_hash, seed)
    fold_list = cv_folds(data, cv_idx, folds, seed)
    sets = feature_sets(data, splits, start_column, n_columns, include_all, pca)
    out_dir = Path(out_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
    show(f"Development rows {len(splits.dev_idx):,} | test rows {len(splits.test_idx):,} | CV rows {len(cv_idx):,}")

    rows = []
    for set_name, (cols, n_components) in sets.items():
        X = data.X[cols].to_numpy(dtype=float)
        for name in models:
            started = time.perf_counter()
            row = {"features": set_name, "n_features": n_components or len(cols), "model": name,
                   **cross_validate(name, X[cv_idx], data.y[cv_idx], fold_list, seed, n_components)}
            pipe = make_pipeline(name, seed, n_components).fit(X[splits.dev_idx], data.y[splits.dev_idx])
            pred = pipe.predict(X[splits.test_idx])
            row.update({f"test_{m}": v for m, v in scores(data.y[splits.test_idx], pred).items()},
                       seconds=round(time.perf_counter() - started, 1))
            if save:
                save_test_outputs(out_dir, set_name, name, data.y[splits.test_idx], pred, data.class_names)
            rows.append(row)
            show(f"{set_name} {name}: CV macro-F1 {row['cv_f1_macro_mean']:.4f}, test macro-F1 "
                 f"{row['test_f1_macro']:.4f}, test accuracy {row['test_accuracy']:.4f}")
    summary = pd.DataFrame(rows)
    if save:
        summary.to_csv(out_dir / "summary.csv", index=False)
    return summary
