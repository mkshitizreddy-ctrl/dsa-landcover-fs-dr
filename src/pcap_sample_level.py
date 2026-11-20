from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits

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
