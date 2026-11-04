import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "pcap_grid_v1"
QUICK_CHECK_DIR = ROOT / "results" / "pcap_grid_quick_check"
SCALERS = ("Standard", "MinMax", "Normalizer")
REDUCTIONS = ("PCA", "KernelPCA", "LDA")
CLASSIFIERS = ("RF", "XGB", "DT", "SVM")
SINGLE_THREADED = {"DT", "SVM"}
CONFIG = ["scaler", "reduction", "k", "classifier"]
KEY = CONFIG + ["fold"]


def make_scaler(name):
    if name == "Standard":
        return StandardScaler()
    if name == "MinMax":
        return MinMaxScaler()
    if name == "Normalizer":
        return Normalizer(norm="l2")
    raise ValueError(f"Unknown scaler {name!r}")


def make_classifier(name, seed=42):
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                             eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(f"Unknown classifier {name!r}")


def k_values(method, n_features, n_classes, max_k=36, quick_check=False):
    top = min(max_k, n_classes - 1) if method == "LDA" else min(max_k, n_features)
    if quick_check:
        return sorted({k for k in (2, 10, top) if k <= top})
    return list(range(1, top + 1))


def fit_reducer(method, Z_train, y_train, n_components, seed=42, kpca_max_train=8000, kpca_solver="arpack"):
    if method == "PCA":
        return PCA(n_components=n_components, svd_solver="full").fit(Z_train)
    if method == "LDA":
        return LinearDiscriminantAnalysis(n_components=n_components, solver="svd").fit(Z_train, y_train)
    if method == "KernelPCA":
        rows = np.arange(len(Z_train))
        if kpca_max_train and len(rows) > kpca_max_train:
            rows, _ = train_test_split(rows, train_size=kpca_max_train, stratify=y_train, random_state=seed)
        model = KernelPCA(n_components=n_components, kernel="rbf", gamma=1.0 / Z_train.shape[1],
                          eigen_solver=kpca_solver, random_state=seed)
        return model.fit(Z_train[rows])
    raise ValueError(f"Unknown reduction {method!r}")


def transform(reducer, Z, chunk=4000):
    return np.vstack([reducer.transform(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])


def evaluate(clf_name, Z_tr, y_tr, Z_va, y_va, class_ids, seed=42):
    out = {"status": "success", "error": ""}
    try:
        start = time.perf_counter()
        model = make_classifier(clf_name, seed).fit(Z_tr, y_tr)
        out["fit_seconds"] = round(time.perf_counter() - start, 3)
        start = time.perf_counter()
        pred = model.predict(Z_va)
        out["predict_seconds"] = round(time.perf_counter() - start, 3)
        out["accuracy"] = accuracy_score(y_va, pred)
        out["balanced_accuracy"] = balanced_accuracy_score(y_va, pred)
        out["f1_macro"] = f1_score(y_va, pred, labels=class_ids, average="macro", zero_division=0)
        per_class = f1_score(y_va, pred, labels=class_ids, average=None, zero_division=0)
        for c, value in zip(class_ids, per_class):
            out[f"f1_class{c}"] = value
    except Exception as exc:
        out.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    return out


def score_over_k(clf_name, ks, Z_tr, y_tr, Z_va, y_va, class_ids, seed, jobs):
    if clf_name in SINGLE_THREADED and jobs > 1:
        return Parallel(n_jobs=jobs)(
            delayed(evaluate)(clf_name, Z_tr[:, :k], y_tr, Z_va[:, :k], y_va, class_ids, seed) for k in ks)
    return [evaluate(clf_name, Z_tr[:, :k], y_tr, Z_va[:, :k], y_va, class_ids, seed) for k in ks]


def load_done(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=KEY + ["status"])
    df = df[df.status == "success"]
    return {(r.scaler, r.reduction, int(r.k), r.classifier, int(r.fold)) for r in df.itertuples()}


def append_rows(path, rows, fields):
    exists = Path(path).exists() and Path(path).stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != fields:
                raise ValueError(f"Refusing to append a different schema to {path}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(fold_csv, n_folds, class_names, out_csv):
    df = pd.read_csv(fold_csv)
    df = df[df.status == "success"].drop_duplicates(KEY, keep="last")
    metrics = ["f1_macro", "balanced_accuracy", "accuracy"] + [f"f1_class{i}" for i in range(len(class_names))]
    groups = df.groupby(CONFIG)
    summary = groups[metrics].mean().add_suffix("_mean")
    summary["f1_macro_sd"] = groups["f1_macro"].std(ddof=1)
    summary["n_folds"] = groups.size()
    summary["fit_seconds_total"] = groups["fit_seconds"].sum()
    summary = summary.reset_index()
    summary = summary[summary.n_folds == n_folds].rename(
        columns={f"f1_class{i}_mean": f"f1_{name}_mean" for i, name in enumerate(class_names)})
    summary = summary.sort_values("f1_macro_mean", ascending=False, kind="stable")
    summary.to_csv(out_csv, index=False)
    return summary


def check_choices(scalers, reductions, classifiers):
    for chosen, allowed in ((scalers, SCALERS), (reductions, REDUCTIONS), (classifiers, CLASSIFIERS)):
        unknown = [x for x in chosen if x not in allowed]
        if unknown:
            raise ValueError(f"Unknown option(s) {unknown}; choose from {allowed}")


def make_logger(log_path, show):
    def log(message):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        show(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def result_fields(n_classes):
    return (["timestamp_utc", "run_id"] + KEY + ["n_train", "n_valid", "accuracy", "balanced_accuracy", "f1_macro"]
            + [f"f1_class{i}" for i in range(n_classes)]
            + ["fit_seconds", "predict_seconds", "reducer_seconds", "status", "error"])


def score_block(block, todo, Z_tr, y_tr, Z_va, y_va, class_ids, fields, run_id, reducer_seconds, seed, jobs):
    rows = []
    for clf_name, ks in todo.items():
        results = score_over_k(clf_name, ks, Z_tr, y_tr, Z_va, y_va, class_ids, seed, jobs)
        for k, result in zip(ks, results):
            row = {f: "" for f in fields}
            row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id,
                       k=k, classifier=clf_name, n_train=len(y_tr), n_valid=len(y_va),
                       reducer_seconds=reducer_seconds, **block, **result)
            rows.append(row)
    return rows


def block_message(block, rows, n_folds, reducer_seconds, block_seconds, elapsed):
    failed = sum(r["status"] == "failed" for r in rows)
    best = max((r for r in rows if r["status"] == "success"), key=lambda r: r["f1_macro"], default=None)
    best_text = f", best {best['classifier']} k={best['k']} macro-F1 {best['f1_macro']:.4f}" if best else ""
    return (f"fold {block['fold']}/{n_folds} {block['scaler']:10} {block['reduction']:9} {len(rows):3} fits, "
            f"{failed} failed, reducer {reducer_seconds}s, block {block_seconds:.0f}s{best_text} "
            f"| elapsed {elapsed:.0f}s")


def run_grid(data_path=DATA_PATH, splits_path=SPLITS_PATH, scalers=SCALERS, reductions=REDUCTIONS,
             classifiers=CLASSIFIERS, max_k=36, kpca_max_train=8000, jobs=None, seed=42, quick_check=False,
             out_dir=None, show=print):
    check_choices(scalers, reductions, classifiers)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    out_dir = Path(out_dir or (QUICK_CHECK_DIR if quick_check else GRID_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = out_dir / "fold_scores.csv"
    log = make_logger(out_dir / "grid.log", show)

    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X, y = data.X[splits.features].to_numpy(dtype=float), data.y
    n_features, n_classes = X.shape[1], len(data.class_names)
    class_ids = list(range(n_classes))
    folds = splits.folds[:1] if quick_check else splits.folds
    fields = result_fields(n_classes)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    done = load_done(fold_csv)
    n_configs = (sum(len(k_values(r, n_features, n_classes, max_k, quick_check)) for r in reductions)
                 * len(scalers) * len(classifiers))
    log(f"START run_id={run_id} configs={n_configs} folds={len(folds)} fits={n_configs * len(folds)} "
        f"already_done={len(done)} features={n_features} classes={data.class_names} jobs={jobs} "
        f"kpca_max_train={kpca_max_train} quick_check={quick_check} "
        f"sklearn={sklearn.__version__} xgboost={xgboost.__version__}")
    started = time.perf_counter()

    for fold, (tr, va) in enumerate(folds, start=1):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
        A_tr, A_va = imputer.transform(X[tr]), imputer.transform(X[va])
        for scaler_name in scalers:
            scaler = make_scaler(scaler_name).fit(A_tr)
            B_tr, B_va = scaler.transform(A_tr), scaler.transform(A_va)
            for reduction in reductions:
                ks = k_values(reduction, n_features, n_classes, max_k, quick_check)
                todo = {c: [k for k in ks if (scaler_name, reduction, k, c, fold) not in done] for c in classifiers}
                todo = {c: kk for c, kk in todo.items() if kk}
                if not todo:
                    continue
                block_start = time.perf_counter()
                reducer = fit_reducer(reduction, B_tr, y[tr], max(ks), seed=seed, kpca_max_train=kpca_max_train)
                Z_tr, Z_va = transform(reducer, B_tr), transform(reducer, B_va)
                reducer_seconds = round(time.perf_counter() - block_start, 2)
                block = {"fold": fold, "scaler": scaler_name, "reduction": reduction}
                rows = score_block(block, todo, Z_tr, y[tr], Z_va, y[va], class_ids, fields, run_id,
                                   reducer_seconds, seed, jobs)
                append_rows(fold_csv, rows, fields)
                log(block_message(block, rows, len(folds), reducer_seconds,
                                  time.perf_counter() - block_start, time.perf_counter() - started))

    elapsed = time.perf_counter() - started
    summary = summarize(fold_csv, len(folds), data.class_names, out_dir / "grid_summary.csv")
    log(f"END elapsed {elapsed:.0f}s | complete configurations {len(summary)}")
    if quick_check:
        full = 3 * (36 + 36 + 5) * 4 * 4
        log(f"Quick check: {n_configs} fits in {elapsed:.0f}s. Estimated full-grid time: "
            f"{elapsed * full / max(n_configs, 1) / 3600:.1f} h (upper bound)")
    return summary


def best_per(summary, column, metric="f1_macro_mean"):
    ranked = summary.sort_values(metric, ascending=False, kind="stable")
    return ranked.groupby(column, sort=False).head(1)
