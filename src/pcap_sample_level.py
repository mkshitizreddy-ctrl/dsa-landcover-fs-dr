import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_grid import CLASSIFIERS, make_classifier

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "results" / "pcap_sample_level_v1"
STATS = ("mean", "std", "min", "max")


@dataclass
class SampleTables:
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    fold_dev: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    class_names: list


def aggregate_by_hash(X, y, groups, stats=STATS):
    frame = pd.DataFrame(X).reset_index(drop=True)
    features = list(frame.columns)
    frame["Hash"] = np.asarray(groups)
    grouped = frame.groupby("Hash", sort=True)
    table = grouped[features].agg(list(stats))
    table.columns = [f"{feature}_{stat}" for feature, stat in table.columns]
    table["packets"] = grouped.size()
    if "std" in stats:
        single = table["packets"] == 1
        std_columns = [f"{feature}_std" for feature in features]
        table.loc[single, std_columns] = table.loc[single, std_columns].fillna(0.0)
    labels = pd.Series(np.asarray(y), index=frame["Hash"]).groupby(level=0).agg(["min", "max"])
    if (labels["min"] != labels["max"]).any():
        raise ValueError("A hash has packets from more than one family")
    return table, labels["min"].reindex(table.index).to_numpy(dtype=int)


def hash_folds(groups, folds):
    fold_of = {}
    for i, (_, valid) in enumerate(folds):
        for name in np.unique(np.asarray(groups)[valid]):
            if fold_of.setdefault(name, i) != i:
                raise ValueError(f"Hash {name!r} is in more than one validation fold")
    return fold_of


def build_sample_tables(data_path=DATA_PATH, splits_path=SPLITS_PATH, stats=STATS):
    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X = data.X[splits.features]
    parts = []
    for rows in (splits.dev_idx, splits.test_idx):
        parts.append(aggregate_by_hash(X.iloc[rows], data.y[rows], data.groups[rows], stats))
    (X_dev, y_dev), (X_test, y_test) = parts
    if set(X_dev.index) & set(X_test.index):
        raise AssertionError("A hash appears in both development and test samples")
    fold_of = hash_folds(data.groups, splits.folds)
    missing = [name for name in X_dev.index if name not in fold_of]
    if missing:
        raise ValueError(f"{len(missing)} development hashes are in no fold")
    fold_dev = np.array([fold_of[name] for name in X_dev.index])
    return SampleTables(X_dev=X_dev, y_dev=y_dev, fold_dev=fold_dev, X_test=X_test, y_test=y_test,
                        class_names=list(data.class_names))


def make_sample_model(classifier, seed=42):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), make_classifier(classifier, seed))


def score_predictions(y_true, pred, class_names):
    labels = np.arange(len(class_names))
    scores = {"accuracy": accuracy_score(y_true, pred),
              "balanced_accuracy": balanced_accuracy_score(y_true, pred),
              "f1_macro": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)}
    per_family = f1_score(y_true, pred, labels=labels, average=None, zero_division=0)
    scores.update({f"f1_{name}": float(value) for name, value in zip(class_names, per_family)})
    return scores


def cross_validate_samples(tables, classifiers=CLASSIFIERS, seed=42, show=print):
    X, y, folds = tables.X_dev.to_numpy(dtype=float), tables.y_dev, tables.fold_dev
    rows = []
    for classifier in classifiers:
        for fold in np.unique(folds):
            train = folds != fold
            start = time.perf_counter()
            model = make_sample_model(classifier, seed).fit(X[train], y[train])
            seconds = round(time.perf_counter() - start, 3)
            pred = model.predict(X[~train])
            rows.append({"classifier": classifier, "fold": int(fold) + 1, "train_samples": int(train.sum()),
                         "valid_samples": int((~train).sum()), "fit_seconds": seconds,
                         **score_predictions(y[~train], pred, tables.class_names)})
        mean = np.mean([row["f1_macro"] for row in rows if row["classifier"] == classifier])
        show(f"{classifier}: CV macro-F1 {mean:.4f}")
    return pd.DataFrame(rows)


def summarize_cv(fold_scores):
    metrics = [c for c in fold_scores.columns if c in ("accuracy", "balanced_accuracy", "f1_macro")
               or c.startswith("f1_") and c != "f1_macro"]
    grouped = fold_scores.groupby("classifier", sort=False)
    summary = grouped[metrics].mean()
    summary.insert(summary.columns.get_loc("f1_macro") + 1, "f1_macro_sd", grouped["f1_macro"].std(ddof=1))
    summary = summary.sort_values("f1_macro", ascending=False, kind="stable").reset_index()
    return summary


def sample_counts(tables):
    dev = np.bincount(tables.y_dev, minlength=len(tables.class_names))
    test = np.bincount(tables.y_test, minlength=len(tables.class_names))
    return pd.DataFrame({"family": tables.class_names, "dev_samples": dev, "test_samples": test})


def run_sample_cv(data_path=DATA_PATH, splits_path=SPLITS_PATH, classifiers=CLASSIFIERS, seed=42, save=True,
                  out_dir=SAMPLE_DIR, tables=None, show=print):
    tables = build_sample_tables(data_path, splits_path) if tables is None else tables
    show(f"Development samples {len(tables.y_dev):,} | test samples {len(tables.y_test):,} | "
         f"columns {tables.X_dev.shape[1]} | folds {len(np.unique(tables.fold_dev))}")
    fold_scores = cross_validate_samples(tables, classifiers, seed, show)
    summary = summarize_cv(fold_scores)
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_scores.to_csv(out_dir / "cv_folds.csv", index=False)
        summary.to_csv(out_dir / "cv_summary.csv", index=False)
        sample_counts(tables).to_csv(out_dir / "samples.csv", index=False)
        show(f"Saved cv_folds.csv, cv_summary.csv and samples.csv to {out_dir}")
    return summary
