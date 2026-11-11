import time
from pathlib import Path

import numpy as np
import pandas as pd

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_feature_selection import FS_DIR, feature_statistics, heap_bfs_subsets, mi_top_subsets, prepare_fold
from pcap_grid import CLASSIFIERS, GRID_DIR, REDUCTIONS, evaluate, fit_reducer, transform

ROOT = Path(__file__).resolve().parents[1]
GRID_SUMMARY_PATH = GRID_DIR / "grid_summary.csv"
FS_SUMMARY_PATH = FS_DIR / "fs_summary.csv"
WEIGHT_DIR = ROOT / "results" / "pcap_weighting_v1"
WEIGHT_SCHEMES = ("none", "sqrt", "balanced")
CONFIG_COLUMNS = ["family", "scaler", "method", "param", "classifier"]


def class_weights(y, scheme):
    if scheme == "none":
        return None
    classes, counts = np.unique(y, return_counts=True)
    per_class = len(y) / (len(classes) * counts)
    if scheme == "sqrt":
        per_class = np.sqrt(per_class)
    elif scheme != "balanced":
        raise ValueError(f"Unknown weighting scheme {scheme!r}; choose from {WEIGHT_SCHEMES}")
    weights = per_class[np.searchsorted(classes, y)]
    return weights * len(y) / weights.sum()


def best_of(table, metric="f1_macro_mean"):
    return table.sort_values(metric, ascending=False, kind="stable").iloc[0]


def choose_configs(grid, fs_summary):
    configs = []
    for clf in CLASSIFIERS:
        g = best_of(grid[grid.classifier == clf])
        configs.append({"family": "extraction", "scaler": g.scaler, "method": g.reduction, "param": float(g.k),
                        "classifier": clf, "cv_f1_macro": g.f1_macro_mean})
        s = best_of(fs_summary[fs_summary.classifier == clf])
        configs.append({"family": "selection", "scaler": s.scaler, "method": s.method, "param": float(s.param),
                        "classifier": clf, "cv_f1_macro": s.f1_macro_mean})
    return configs


def config_features(config, Z_train, y_train, Z_valid, stats, seed=42, kpca_max_train=8000):
    if config["method"] in REDUCTIONS:
        reducer = fit_reducer(config["method"], Z_train, y_train, int(config["param"]), seed=seed,
                              kpca_max_train=kpca_max_train)
        return transform(reducer, Z_train), transform(reducer, Z_valid)
    scores, C = stats
    if config["method"] == "MI_top":
        k = int(config["param"])
        cols = list(mi_top_subsets(scores, [k])[k])
    else:
        t = float(config["param"])
        cols = list(heap_bfs_subsets(scores, C, [t])[t])
    return Z_train[:, cols], Z_valid[:, cols]


def summarize_weighting(fold_scores, class_names):
    metrics = ["f1_macro", "balanced_accuracy", "accuracy"] + [f"f1_class{i}" for i in range(len(class_names))]
    ok = fold_scores[fold_scores.status == "success"]
    summary = ok.groupby(CONFIG_COLUMNS + ["scheme"], sort=False)[metrics].mean().reset_index()
    return summary.rename(columns={f"f1_class{i}": f"f1_{name}" for i, name in enumerate(class_names)})


def run_class_weighting(data_path=DATA_PATH, splits_path=SPLITS_PATH, grid_path=GRID_SUMMARY_PATH,
                        fs_summary_path=FS_SUMMARY_PATH, schemes=WEIGHT_SCHEMES, kpca_max_train=8000, seed=42,
                        save=True, out_dir=WEIGHT_DIR, show=print):
    unknown = [s for s in schemes if s not in WEIGHT_SCHEMES]
    if unknown:
        raise ValueError(f"Unknown weighting schemes {unknown}; choose from {WEIGHT_SCHEMES}")
    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X, y = data.X[splits.features].to_numpy(dtype=float), data.y
    class_ids = list(range(len(data.class_names)))
    configs = choose_configs(pd.read_csv(grid_path), pd.read_csv(fs_summary_path))
    show(f"{len(configs)} configurations x {len(schemes)} weighting schemes x {len(splits.folds)} folds")

    started, rows = time.perf_counter(), []
    for fold, (tr, va) in enumerate(splits.folds, start=1):
        for scaler in sorted({c["scaler"] for c in configs}):
            Z_train, Z_valid = prepare_fold(X, tr, va, scaler)
            stats = feature_statistics(Z_train, y[tr])
            for config in (c for c in configs if c["scaler"] == scaler):
                F_train, F_valid = config_features(config, Z_train, y[tr], Z_valid, stats, seed, kpca_max_train)
                for scheme in schemes:
                    out = evaluate(config["classifier"], F_train, y[tr], F_valid, y[va], class_ids, seed,
                                   sample_weight=class_weights(y[tr], scheme))
                    rows.append({**{c: config[c] for c in CONFIG_COLUMNS}, "scheme": scheme, "fold": fold,
                                 "n_features": F_train.shape[1], **out})
        show(f"fold {fold}/{len(splits.folds)} done | elapsed {time.perf_counter() - started:.0f}s")

    fold_scores = pd.DataFrame(rows)
    summary = summarize_weighting(fold_scores, data.class_names)
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_scores.to_csv(out_dir / "fold_scores.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        show(f"Saved {out_dir / 'summary.csv'}")
    return summary
