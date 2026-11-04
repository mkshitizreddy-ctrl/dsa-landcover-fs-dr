import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pcap.csv"
SPLITS_PATH = ROOT / "data" / "pcap_splits.npz"
NON_FEATURES = ["Hash", "Arch", "MalwareFamily"]
DROP_CLASSES = ["Agent", "Tsunami"]


@dataclass
class PcapData:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    class_names: list
    dropped_rows: int


@dataclass
class Splits:
    dev_idx: np.ndarray
    test_idx: np.ndarray
    sample_idx: np.ndarray
    folds: list
    features: list
    meta: dict = field(default_factory=dict)


def load_pcap(path=DATA_PATH, drop_classes=DROP_CLASSES):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = set(NON_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = df["MalwareFamily"].astype(str).str.strip()
    hashes = df["Hash"].astype(str).str.strip()
    if labels.eq("").any() or hashes.eq("").any():
        raise ValueError("Empty MalwareFamily or Hash values found.")

    keep = ~labels.isin(drop_classes)
    X = (df.loc[keep].drop(columns=NON_FEATURES)
           .apply(pd.to_numeric, errors="raise")
           .replace([np.inf, -np.inf], np.nan)
           .reset_index(drop=True))
    class_names = sorted(labels[keep].unique())
    class_id = {name: i for i, name in enumerate(class_names)}
    y = labels[keep].map(class_id).to_numpy(dtype=int)
    groups = hashes[keep].to_numpy()
    return PcapData(X=X, y=y, groups=groups, class_names=class_names, dropped_rows=int((~keep).sum()))


def check_all_classes(y_part, n_classes, part_name):
    missing = set(range(n_classes)) - set(np.unique(y_part))
    if missing:
        raise ValueError(f"{part_name} is missing class ids {sorted(missing)}")


def check_no_shared_groups(groups, rows_a, rows_b, part_name):
    if set(groups[rows_a]) & set(groups[rows_b]):
        raise AssertionError(f"A hash appears on both sides of {part_name}")


def cap_rows_per_group(rows, groups, cap, seed):
    if cap <= 0:
        return np.sort(rows)
    order = np.random.default_rng(seed).permutation(rows)
    position_in_group = pd.Series(groups[order]).groupby(groups[order]).cumcount().to_numpy()
    return np.sort(order[position_in_group < cap])


def make_splits(data, seed=42, test_folds=5, cv_folds=4, rows_per_hash=5):
    y, groups = data.y, data.groups
    n_classes = len(data.class_names)

    holdout = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(holdout.split(np.arange(len(y)), y, groups))
    check_all_classes(y[dev_idx], n_classes, "Development set")
    check_all_classes(y[test_idx], n_classes, "Test set")
    check_no_shared_groups(groups, dev_idx, test_idx, "the test split")

    sample_idx = cap_rows_per_group(dev_idx, groups, rows_per_hash, seed)

    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = []
    for i, (train, valid) in enumerate(cv.split(sample_idx, y[sample_idx], groups[sample_idx]), start=1):
        train_idx, valid_idx = sample_idx[train], sample_idx[valid]
        check_all_classes(y[train_idx], n_classes, f"Fold {i} training")
        check_all_classes(y[valid_idx], n_classes, f"Fold {i} validation (try fewer folds)")
        check_no_shared_groups(groups, train_idx, valid_idx, f"fold {i}")
        folds.append((train_idx, valid_idx))

    dev_X = data.X.iloc[dev_idx]
    features = [c for c in data.X.columns if dev_X[c].nunique(dropna=True) > 1]
    meta = {
        "n_rows": int(len(y)), "dropped_rows": data.dropped_rows, "class_names": data.class_names,
        "seed": seed, "test_folds": test_folds, "cv_folds": cv_folds, "rows_per_hash": rows_per_hash,
        "constant_columns_dropped": [c for c in data.X.columns if c not in features],
    }
    return Splits(dev_idx=np.sort(dev_idx), test_idx=np.sort(test_idx), sample_idx=sample_idx,
                  folds=folds, features=features, meta=meta)


def save_splits(splits, path=SPLITS_PATH):
    arrays = {"dev_idx": splits.dev_idx, "test_idx": splits.test_idx, "sample_idx": splits.sample_idx}
    for i, (train_idx, valid_idx) in enumerate(splits.folds):
        arrays[f"fold{i}_train"], arrays[f"fold{i}_valid"] = train_idx, valid_idx
    meta = {**splits.meta, "features": splits.features, "n_folds": len(splits.folds)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_splits(path=SPLITS_PATH, n_rows=None):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["meta"]))
        folds = [(saved[f"fold{i}_train"], saved[f"fold{i}_valid"]) for i in range(meta["n_folds"])]
        splits = Splits(dev_idx=saved["dev_idx"], test_idx=saved["test_idx"], sample_idx=saved["sample_idx"],
                        folds=folds, features=meta.pop("features"), meta=meta)
    if n_rows is not None and n_rows != meta["n_rows"]:
        raise ValueError(f"Splits were made for {meta['n_rows']} rows, data has {n_rows}; "
                         "create the splits again")
    return splits


def split_table(data, splits):
    names = np.array(data.class_names)

    def rows_per_class(rows):
        return pd.Series(names[data.y[rows]]).value_counts()

    def hashes_per_class(rows):
        return pd.Series(data.groups[rows]).groupby(names[data.y[rows]]).nunique()

    return pd.DataFrame({
        "dev_rows": rows_per_class(splits.dev_idx), "dev_hashes": hashes_per_class(splits.dev_idx),
        "test_rows": rows_per_class(splits.test_idx), "test_hashes": hashes_per_class(splits.test_idx),
        "sample_rows": rows_per_class(splits.sample_idx),
    }).fillna(0).astype(int)


def summarize(data, splits):
    lines = [
        f"Rows {splits.meta['n_rows']:,} (removed {splits.meta['dropped_rows']} rows of {DROP_CLASSES})",
        f"Features {len(splits.features)} (constant columns removed: {splits.meta['constant_columns_dropped']})",
        f"Development {len(splits.dev_idx):,} | test {len(splits.test_idx):,} "
        f"({len(splits.test_idx) / splits.meta['n_rows']:.1%}) | grid sample {len(splits.sample_idx):,}",
        split_table(data, splits).to_string(),
    ]
    for i, (train_idx, valid_idx) in enumerate(splits.folds, start=1):
        lines.append(f"Fold {i}: train {len(train_idx):,} rows, valid {len(valid_idx):,} rows")
    return "\n".join(lines)


def create_splits(data_path=DATA_PATH, splits_path=SPLITS_PATH, rows_per_hash=5, cv_folds=4, seed=42):
    data = load_pcap(data_path)
    splits = make_splits(data, seed=seed, cv_folds=cv_folds, rows_per_hash=rows_per_hash)
    save_splits(splits, splits_path)
    load_splits(splits_path, n_rows=len(data.y))
    return data, splits
