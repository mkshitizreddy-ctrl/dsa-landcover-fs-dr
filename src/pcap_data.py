"""Shared data preparation for every pcap.csv experiment.

One module decides which rows go where, and saves the row indices to a file,
so every script uses identical splits (and results from different runs can be
merged or looked up safely).

Rules:
  * Hash, Arch and MalwareFamily are not features.
  * Classes with too few hashes for a grouped split (Agent, Tsunami) are dropped.
  * inf is turned into NaN; imputation is left to each model pipeline, so it
    is always fitted on training rows only.
  * Test holdout: first fold of StratifiedGroupKFold(5) over Hash (~20% of rows).
    Never used until the final evaluation.
  * Grid sample: at most `rows_per_hash` rows from each development hash.
  * CV: StratifiedGroupKFold(4) over Hash on the grid sample.
  * Constant columns are found on development rows only.
All indices saved are positions in the cleaned table returned by load_pcap().

Command line (from the repository root):
    python src\\pcap_data.py --data data\\pcap.csv --out data\\pcap_splits.npz
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

NON_FEATURES = ["Hash", "Arch", "MalwareFamily"]
DROP_CLASSES = ["Agent", "Tsunami"]


@dataclass
class PcapData:
    X: pd.DataFrame            # Shape [N, 39]; NaN where values were missing or infinite
    y: np.ndarray              # Shape [N]; class ids 0..C-1
    groups: np.ndarray         # Shape [N]; Hash strings
    class_names: list
    dropped_rows: int


@dataclass
class Splits:
    dev_idx: np.ndarray        # rows available for model development
    test_idx: np.ndarray       # untouched holdout
    sample_idx: np.ndarray     # grid sample, a subset of dev_idx
    folds: list                # [(train_idx, valid_idx), ...] over sample_idx rows
    features: list             # non-constant feature columns (decided on dev rows)
    meta: dict = field(default_factory=dict)


def load_pcap(path, drop_classes=DROP_CLASSES):
    """Read and clean the CSV. Raises on missing columns or non-numeric features."""
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
    dropped = int((~keep).sum())
    X = (df.loc[keep].drop(columns=NON_FEATURES)
           .apply(pd.to_numeric, errors="raise")
           .replace([np.inf, -np.inf], np.nan)
           .reset_index(drop=True))
    class_names = sorted(labels[keep].unique())
    lookup = {name: i for i, name in enumerate(class_names)}
    y = labels[keep].map(lookup).to_numpy(dtype=int)
    groups = hashes[keep].to_numpy()
    return PcapData(X=X, y=y, groups=groups, class_names=class_names, dropped_rows=dropped)


def _check(y_part, g_part, n_classes, what):
    missing = set(range(n_classes)) - set(np.unique(y_part))
    if missing:
        raise ValueError(f"{what} is missing class ids {sorted(missing)}")


def _cap_per_group(rows, groups, cap, seed):
    """Keep at most `cap` randomly chosen rows per group from `rows`."""
    if cap <= 0:
        return np.sort(rows)
    rng = np.random.default_rng(seed)
    order = rng.permutation(rows)
    rank = pd.Series(groups[order]).groupby(groups[order]).cumcount().to_numpy()
    return np.sort(order[rank < cap])


def make_splits(data, seed=42, test_folds=5, cv_folds=4, rows_per_hash=5):
    y, groups = data.y, data.groups
    n_classes = len(data.class_names)
    all_rows = np.arange(len(y))

    holdout = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(holdout.split(all_rows, y, groups))
    _check(y[dev_idx], groups[dev_idx], n_classes, "Development set")
    _check(y[test_idx], groups[test_idx], n_classes, "Test set")
    if set(groups[dev_idx]) & set(groups[test_idx]):
        raise AssertionError("A hash appears in both development and test rows")

    sample_idx = _cap_per_group(dev_idx, groups, rows_per_hash, seed)

    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = []
    for i, (tr, va) in enumerate(cv.split(sample_idx, y[sample_idx], groups[sample_idx]), start=1):
        tr_idx, va_idx = sample_idx[tr], sample_idx[va]
        _check(y[tr_idx], groups[tr_idx], n_classes, f"Fold {i} training")
        _check(y[va_idx], groups[va_idx], n_classes, f"Fold {i} validation (try fewer folds)")
        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise AssertionError(f"A hash appears in both training and validation of fold {i}")
        folds.append((tr_idx, va_idx))

    dev_X = data.X.iloc[dev_idx]
    features = [c for c in data.X.columns if dev_X[c].nunique(dropna=True) > 1]

    meta = {
        "n_rows": int(len(y)), "dropped_rows": data.dropped_rows, "class_names": data.class_names,
        "seed": seed, "test_folds": test_folds, "cv_folds": cv_folds, "rows_per_hash": rows_per_hash,
        "constant_columns_dropped": [c for c in data.X.columns if c not in features],
    }
    return Splits(dev_idx=np.sort(dev_idx), test_idx=np.sort(test_idx), sample_idx=sample_idx,
                  folds=folds, features=features, meta=meta)


def save_splits(splits, path):
    arrays = {"dev_idx": splits.dev_idx, "test_idx": splits.test_idx, "sample_idx": splits.sample_idx}
    for i, (tr, va) in enumerate(splits.folds):
        arrays[f"fold{i}_train"], arrays[f"fold{i}_valid"] = tr, va
    meta = {**splits.meta, "features": splits.features, "n_folds": len(splits.folds)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_splits(path, n_rows=None):
    """Load saved splits; with n_rows, check they belong to this cleaned table."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        folds = [(z[f"fold{i}_train"], z[f"fold{i}_valid"]) for i in range(meta["n_folds"])]
        splits = Splits(dev_idx=z["dev_idx"], test_idx=z["test_idx"], sample_idx=z["sample_idx"],
                        folds=folds, features=meta.pop("features"), meta=meta)
    if n_rows is not None and n_rows != meta["n_rows"]:
        raise ValueError(f"Splits were made for {meta['n_rows']} rows, data has {n_rows}; "
                         "regenerate them with pcap_data.py")
    return splits


def summarize(data, splits):
    names = np.array(data.class_names)
    def counts(rows):
        return pd.Series(names[data.y[rows]]).value_counts()
    def hashes(rows):
        return pd.Series(data.groups[rows]).groupby(names[data.y[rows]]).nunique()
    table = pd.DataFrame({"dev_rows": counts(splits.dev_idx), "dev_hashes": hashes(splits.dev_idx),
                          "test_rows": counts(splits.test_idx), "test_hashes": hashes(splits.test_idx),
                          "sample_rows": counts(splits.sample_idx)}).fillna(0).astype(int)
    lines = [f"rows {splits.meta['n_rows']:,} (dropped {splits.meta['dropped_rows']} of {DROP_CLASSES})",
             f"features {len(splits.features)} (constant dropped: {splits.meta['constant_columns_dropped']})",
             f"dev {len(splits.dev_idx):,} | test {len(splits.test_idx):,} "
             f"({len(splits.test_idx) / splits.meta['n_rows']:.1%}) | grid sample {len(splits.sample_idx):,}",
             table.to_string()]
    for i, (tr, va) in enumerate(splits.folds, start=1):
        lines.append(f"fold {i}: train {len(tr):,} rows, valid {len(va):,} rows")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Create and save the shared pcap splits.")
    p.add_argument("--data", default="data/pcap.csv")
    p.add_argument("--out", default="data/pcap_splits.npz")
    p.add_argument("--rows-per-hash", type=int, default=5)
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = load_pcap(args.data)
    splits = make_splits(data, seed=args.seed, cv_folds=args.cv_folds, rows_per_hash=args.rows_per_hash)
    save_splits(splits, args.out)
    reloaded = load_splits(args.out, n_rows=len(data.y))
    assert all((a == b).all() for a, b in zip(reloaded.folds[0], splits.folds[0]))
    print(summarize(data, splits))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
