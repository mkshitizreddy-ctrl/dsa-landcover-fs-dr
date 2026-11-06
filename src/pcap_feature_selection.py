import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer

from graph import build_corr_graph
from mi import relevance_scores
from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_grid import CLASSIFIERS, GRID_DIR, REDUCTIONS, SCALERS, SINGLE_THREADED, append_rows, check_choices, \
    evaluate, make_logger, make_scaler
from selection import select_features
from sorting import merge_sort

ROOT = Path(__file__).resolve().parents[1]
FS_DIR = ROOT / "results" / "pcap_fs_v1"
FS_QUICK_CHECK_DIR = ROOT / "results" / "pcap_fs_quick_check"
GRID_SUMMARY_PATH = GRID_DIR / "grid_summary.csv"
METHODS = ("MI_top", "HeapBFS")
THRESHOLDS = tuple(i / 100 for i in range(101))
QUICK_K = (2, 10, 36)
QUICK_THRESHOLDS = (0.5, 0.9, 1.0)
KEY = ["scaler", "method", "param", "classifier", "fold"]


def prepare_fold(X, train_rows, valid_rows, scaler_name="Standard"):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[train_rows])
    A_train, A_valid = imputer.transform(X[train_rows]), imputer.transform(X[valid_rows])
    scaler = make_scaler(scaler_name).fit(A_train)
    return scaler.transform(A_train), scaler.transform(A_valid)


def feature_statistics(Z_train, y_train, bins=10):
    scores = relevance_scores(Z_train, y_train, bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.abs(np.corrcoef(Z_train, rowvar=False)), nan=0.0)
    return scores, C


def mi_top_subsets(scores, ks):
    order = merge_sort(list(range(len(scores))), key=lambda j: scores[j], reverse=True)
    return {k: tuple(sorted(order[:k])) for k in ks}


def heap_bfs_subsets(scores, C, thresholds):
    return {t: tuple(sorted(select_features(scores, build_corr_graph(C, t)))) for t in thresholds}


def score_subsets(clf_name, subsets, Z_train, y_train, Z_valid, y_valid, class_ids, seed, jobs):
    unique = sorted(set(subsets))
    if clf_name in SINGLE_THREADED and jobs > 1:
        results = Parallel(n_jobs=jobs)(
            delayed(evaluate)(clf_name, Z_train[:, list(cols)], y_train, Z_valid[:, list(cols)], y_valid,
                              class_ids, seed) for cols in unique)
    else:
        results = [evaluate(clf_name, Z_train[:, list(cols)], y_train, Z_valid[:, list(cols)], y_valid,
                            class_ids, seed) for cols in unique]
    return dict(zip(unique, results))


def result_fields(n_classes):
    return (["timestamp_utc", "run_id"] + KEY + ["k", "features", "n_train", "n_valid",
                                                 "accuracy", "balanced_accuracy", "f1_macro"]
            + [f"f1_class{i}" for i in range(n_classes)] + ["fit_seconds", "predict_seconds", "status", "error"])


def load_done(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=KEY + ["status"])
    df = df[df.status == "success"]
    return {(r.scaler, r.method, float(r.param), r.classifier, int(r.fold)) for r in df.itertuples()}


def fold_rows(block, clf_name, candidates, scored, feature_names, fields, run_id, n_train, n_valid):
    rows = []
    for (method, param), cols in candidates.items():
        row = {f: "" for f in fields}
        row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id,
                   method=method, param=param, classifier=clf_name, k=len(cols), **block,
                   features=";".join(feature_names[j] for j in cols), n_train=n_train, n_valid=n_valid,
                   **scored[cols])
        rows.append(row)
    return rows


def summarize(fold_csv, n_folds, out_csv):
    df = pd.read_csv(fold_csv)
    df = df[df.status == "success"].drop_duplicates(KEY, keep="last")
    groups = df.groupby(["scaler", "method", "param", "classifier"])
    summary = groups[["f1_macro", "balanced_accuracy", "accuracy", "k"]].mean().add_suffix("_mean")
    summary["f1_macro_sd"] = groups["f1_macro"].std(ddof=1)
    summary["k_min"] = groups["k"].min()
    summary["k_max"] = groups["k"].max()
    summary["n_folds"] = groups.size()
    summary = summary.reset_index()
    summary = summary[summary.n_folds == n_folds]
    summary = summary.sort_values("f1_macro_mean", ascending=False, kind="stable")
    summary.to_csv(out_csv, index=False)
    return summary


def best_row(table):
    return table.sort_values("f1_macro_mean", ascending=False, kind="stable").iloc[0]


def compare_with_grid(summary, grid_path=GRID_SUMMARY_PATH):
    grid = pd.read_csv(grid_path)
    rows = []
    for scaler in summary.scaler.unique():
        for clf in summary.classifier.unique():
            for method in METHODS:
                part = summary[(summary.scaler == scaler) & (summary.method == method) & (summary.classifier == clf)]
                if len(part):
                    best = best_row(part)
                    setting = f"t = {best.param:.2f}" if method == "HeapBFS" else f"k = {int(best.param)}"
                    rows.append({"scaler": scaler, "classifier": clf, "method": method, "setting": setting,
                                 "features_or_components": round(best.k_mean, 1),
                                 "cv_f1_macro": best.f1_macro_mean, "cv_accuracy": best.accuracy_mean})
            for reduction in REDUCTIONS:
                part = grid[(grid.scaler == scaler) & (grid.reduction == reduction) & (grid.classifier == clf)]
                if len(part):
                    best = best_row(part)
                    rows.append({"scaler": scaler, "classifier": clf, "method": reduction,
                                 "setting": f"k = {int(best.k)}", "features_or_components": int(best.k),
                                 "cv_f1_macro": best.f1_macro_mean, "cv_accuracy": best.accuracy_mean})
    return pd.DataFrame(rows)


def best_per_method(comparison):
    ranked = comparison.sort_values("cv_f1_macro", ascending=False, kind="stable")
    return ranked.groupby("method", sort=False).head(1).reset_index(drop=True)


def run_pcap_feature_selection(data_path=DATA_PATH, splits_path=SPLITS_PATH, scalers=SCALERS,
                               classifiers=CLASSIFIERS, thresholds=THRESHOLDS, max_k=36, mi_bins=10, jobs=None,
                               seed=42, quick_check=False, out_dir=None, show=print):
    check_choices(scalers, (), classifiers)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    out_dir = Path(out_dir or (FS_QUICK_CHECK_DIR if quick_check else FS_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = out_dir / "fold_scores.csv"
    log = make_logger(out_dir / "fs.log", show)

    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X, y = data.X[splits.features].to_numpy(dtype=float), data.y
    feature_names = list(splits.features)
    n_features, n_classes = X.shape[1], len(data.class_names)
    class_ids = list(range(n_classes))
    folds = splits.folds[:1] if quick_check else splits.folds
    ks = [k for k in QUICK_K if k <= n_features] if quick_check else list(range(1, min(max_k, n_features) + 1))
    thresholds = QUICK_THRESHOLDS if quick_check else tuple(thresholds)
    fields = result_fields(n_classes)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    done = load_done(fold_csv)
    log(f"START run_id={run_id} folds={len(folds)} scalers={list(scalers)} classifiers={list(classifiers)} "
        f"k values={len(ks)} "
        f"thresholds={len(thresholds)} features={n_features} jobs={jobs} quick_check={quick_check}")
    started = time.perf_counter()

    mi_rows = []
    for fold, (tr, va) in enumerate(folds, start=1):
        for scaler_name in scalers:
            Z_train, Z_valid = prepare_fold(X, tr, va, scaler_name)
            scores, C = feature_statistics(Z_train, y[tr], bins=mi_bins)
            mi_rows += [{"fold": fold, "scaler": scaler_name, "feature": name, "mi": score}
                        for name, score in zip(feature_names, scores)]
            candidates = {("MI_top", float(k)): cols for k, cols in mi_top_subsets(scores, ks).items()}
            candidates.update({("HeapBFS", float(t)): cols
                               for t, cols in heap_bfs_subsets(scores, C, thresholds).items()})
            log(f"fold {fold} {scaler_name}: {len(candidates)} settings, "
                f"{len(set(candidates.values()))} distinct feature subsets")
            block = {"fold": fold, "scaler": scaler_name}
            for clf_name in classifiers:
                todo = {key: cols for key, cols in candidates.items()
                        if (scaler_name, *key, clf_name, fold) not in done}
                if not todo:
                    continue
                block_start = time.perf_counter()
                scored = score_subsets(clf_name, todo.values(), Z_train, y[tr], Z_valid, y[va], class_ids, seed,
                                       jobs)
                rows = fold_rows(block, clf_name, todo, scored, feature_names, fields, run_id, len(tr), len(va))
                append_rows(fold_csv, rows, fields)
                best = max((r for r in rows if r["status"] == "success"), key=lambda r: r["f1_macro"], default=None)
                best_text = f", best {best['method']} {best['param']} macro-F1 {best['f1_macro']:.4f}" if best else ""
                log(f"fold {fold}/{len(folds)} {scaler_name:10} {clf_name:3} {len(scored)} fits{best_text} "
                    f"| block {time.perf_counter() - block_start:.0f}s | elapsed {time.perf_counter() - started:.0f}s")

    pd.DataFrame(mi_rows).to_csv(out_dir / "mi_scores.csv", index=False)
    summary = summarize(fold_csv, len(folds), out_dir / "fs_summary.csv")
    if GRID_SUMMARY_PATH.exists():
        compare_with_grid(summary).to_csv(out_dir / "comparison_with_grid.csv", index=False)
    log(f"END elapsed {time.perf_counter() - started:.0f}s | complete settings {len(summary)}")
    return summary
