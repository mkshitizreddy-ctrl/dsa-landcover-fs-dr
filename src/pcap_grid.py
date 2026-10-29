"""Exhaustive grid on the shared pcap splits: scaler x reduction x k x classifier.

Search space (course reference pipeline; KNN excluded):
  scalers      StandardScaler, MinMaxScaler, Normalizer(L2)
  reductions   PCA (k = 1..36), KernelPCA rbf (k = 1..36), LDA (k = 1..5)
  classifiers  RandomForest, XGBoost, DecisionTree, SVM (rbf)
  -> 3 x (36 + 36 + 5) x 4 = 924 configurations, each scored on 4 grouped folds.

Pipeline per fold: median imputation -> scaler -> reduction -> classifier, all
fitted on the fold's training rows only (rows come from data/pcap_splits.npz).

Speed-ups that do not change results:
  * Each reduction is fitted ONCE per fold and scaler with the most components
    needed; a configuration with k components uses the first k columns. PCA,
    KernelPCA and LDA order their components, so this equals fitting with k.
  * Single-threaded models (DT, SVM) run the different k values in parallel
    processes; RF and XGBoost use all cores themselves.
KernelPCA stores an n x n kernel matrix, so it is fitted on at most
--kpca-max-train training rows (stratified); every training row is then
projected with it and the classifier still trains on all of them.

Results append to <out>/fold_scores.csv after every block, and a rerun skips
configurations already scored, so an interrupted run can simply be restarted.
<out>/grid_summary.csv has one row per configuration (mean and SD over folds).

Run from the repository root:
    python src\\pcap_grid.py --smoke     (fold 1, a few k values: timing check)
    python src\\pcap_grid.py             (full grid)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcap_data import load_pcap, load_splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCALERS = ("Standard", "MinMax", "Normalizer")
REDUCTIONS = ("PCA", "KernelPCA", "LDA")
CLASSIFIERS = ("RF", "XGB", "DT", "SVM")
PARALLEL_SAFE = {"DT", "SVM"}          # single-threaded: parallelise over k
KEY = ["scaler", "reduction", "k", "classifier", "fold"]


# ------------------------------------------------------------------ factories
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


def k_values(method, n_features, n_classes, max_k=36, smoke=False):
    """Component counts searched for one reduction method."""
    top = min(max_k, n_classes - 1) if method == "LDA" else min(max_k, n_features)
    if smoke:
        return sorted({k for k in (2, 10, top) if k <= top})
    return list(range(1, top + 1))


def fit_reducer(method, Z_train, y_train, n_components, seed=42,
                kpca_max_train=8000, kpca_solver="arpack"):
    """Fit one reduction with n_components; callers slice the first k columns.

    Z_train: Shape [N_train, D]. Returns the fitted reducer.
    """
    if method == "PCA":
        return PCA(n_components=n_components, svd_solver="full").fit(Z_train)
    if method == "LDA":
        return LinearDiscriminantAnalysis(n_components=n_components, solver="svd").fit(Z_train, y_train)
    if method == "KernelPCA":
        rows = np.arange(len(Z_train))
        if kpca_max_train and len(rows) > kpca_max_train:
            rows, _ = train_test_split(rows, train_size=kpca_max_train, stratify=y_train,
                                       random_state=seed)
        model = KernelPCA(n_components=n_components, kernel="rbf", gamma=1.0 / Z_train.shape[1],
                          eigen_solver=kpca_solver, random_state=seed)
        return model.fit(Z_train[rows])
    raise ValueError(f"Unknown reduction {method!r}")


def transform(reducer, Z, chunk=4000):
    """Chunked transform keeps the KernelPCA cross-kernel small in memory."""
    return np.vstack([reducer.transform(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])


def evaluate(clf_name, Z_tr, y_tr, Z_va, y_va, class_ids, seed=42):
    """Fit one classifier and score it. Never raises: failures are returned."""
    out = {"status": "success", "error": ""}
    try:
        t0 = time.perf_counter()
        model = make_classifier(clf_name, seed).fit(Z_tr, y_tr)
        out["fit_seconds"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        pred = model.predict(Z_va)
        out["predict_seconds"] = round(time.perf_counter() - t0, 3)
        out["accuracy"] = accuracy_score(y_va, pred)
        out["balanced_accuracy"] = balanced_accuracy_score(y_va, pred)
        out["f1_macro"] = f1_score(y_va, pred, labels=class_ids, average="macro", zero_division=0)
        for c, v in zip(class_ids, f1_score(y_va, pred, labels=class_ids, average=None, zero_division=0)):
            out[f"f1_class{c}"] = v
    except Exception as exc:
        out.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    return out


# ------------------------------------------------------------------------- io
def load_done(path):
    """Configurations already scored successfully (for resuming)."""
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=KEY + ["status"])
    df = df[df.status == "success"]
    return {(r.scaler, r.reduction, int(r.k), r.classifier, int(r.fold)) for r in df.itertuples()}


def append_rows(path, rows, fields):
    exists = Path(path).exists() and Path(path).stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != fields:
            raise ValueError(f"Refusing to append a different schema to {path}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def summarize(fold_csv, n_folds, class_names, out_csv):
    df = pd.read_csv(fold_csv)
    df = df[df.status == "success"].drop_duplicates(KEY, keep="last")
    cfg = ["scaler", "reduction", "k", "classifier"]
    metrics = ["f1_macro", "balanced_accuracy", "accuracy"] + [f"f1_class{i}" for i in range(len(class_names))]
    g = df.groupby(cfg)
    summary = g[metrics].mean().add_suffix("_mean")
    summary["f1_macro_sd"] = g["f1_macro"].std(ddof=1)
    summary["n_folds"] = g.size()
    summary["fit_seconds_total"] = g["fit_seconds"].sum()
    summary = summary.reset_index()
    summary = summary[summary.n_folds == n_folds].rename(
        columns={f"f1_class{i}_mean": f"f1_{name}_mean" for i, name in enumerate(class_names)})
    summary = summary.sort_values("f1_macro_mean", ascending=False, kind="stable")
    summary.to_csv(out_csv, index=False)
    return summary


# ----------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "pcap.csv"))
    p.add_argument("--splits", default=str(ROOT / "data" / "pcap_splits.npz"))
    p.add_argument("--scalers", default=",".join(SCALERS))
    p.add_argument("--reductions", default=",".join(REDUCTIONS))
    p.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    p.add_argument("--max-k", type=int, default=36)
    p.add_argument("--kpca-max-train", type=int, default=8000, help="0 = fit KernelPCA on all training rows")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="parallel processes for DT/SVM over k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="fold 1 only, k in {2, 10, max}: timing check")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    reductions = [r.strip() for r in args.reductions.split(",") if r.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    for name, allowed in ((scalers, SCALERS), (reductions, REDUCTIONS), (classifiers, CLASSIFIERS)):
        bad = [x for x in name if x not in allowed]
        if bad:
            raise ValueError(f"Unknown option(s) {bad}; choose from {allowed}")
    out = Path(args.out or ROOT / "results" / ("pcap_grid_smoke" if args.smoke else "pcap_grid_v1"))
    out.mkdir(parents=True, exist_ok=True)
    fold_csv, log_path = out / "fold_scores.csv", out / "grid.log"

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    data = load_pcap(args.data)
    splits = load_splits(args.splits, n_rows=len(data.y))
    X = data.X[splits.features].to_numpy(dtype=float)              # Shape [N, D]
    y = data.y
    D, n_classes = X.shape[1], len(data.class_names)
    class_ids = list(range(n_classes))
    folds = splits.folds[:1] if args.smoke else splits.folds

    fields = (["timestamp_utc", "run_id"] + KEY + ["n_train", "n_valid", "accuracy", "balanced_accuracy",
              "f1_macro"] + [f"f1_class{i}" for i in class_ids] +
              ["fit_seconds", "predict_seconds", "reducer_seconds", "status", "error"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    done = load_done(fold_csv)
    n_cfg = sum(len(k_values(r, D, n_classes, args.max_k, args.smoke)) for r in reductions) \
        * len(scalers) * len(classifiers)
    import sklearn, xgboost
    log(f"START run_id={run_id} configs={n_cfg} folds={len(folds)} fits={n_cfg * len(folds)} "
        f"already_done={len(done)} features={D} classes={data.class_names} jobs={args.jobs} "
        f"kpca_max_train={args.kpca_max_train} smoke={args.smoke} "
        f"sklearn={sklearn.__version__} xgboost={xgboost.__version__}")
    started = time.perf_counter()

    for fi, (tr, va) in enumerate(folds, start=1):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
        A_tr, A_va = imputer.transform(X[tr]), imputer.transform(X[va])
        for sname in scalers:
            scaler = make_scaler(sname).fit(A_tr)
            B_tr, B_va = scaler.transform(A_tr), scaler.transform(A_va)      # Shape [N, D]
            for red in reductions:
                ks = k_values(red, D, n_classes, args.max_k, args.smoke)
                todo = {c: [k for k in ks if (sname, red, k, c, fi) not in done] for c in classifiers}
                if not any(todo.values()):
                    continue
                block_start = time.perf_counter()
                reducer = fit_reducer(red, B_tr, y[tr], max(ks), seed=args.seed,
                                      kpca_max_train=args.kpca_max_train)
                Z_tr, Z_va = transform(reducer, B_tr), transform(reducer, B_va)   # Shape [N, max k]
                reducer_seconds = round(time.perf_counter() - block_start, 2)

                rows = []
                for c, kk in todo.items():
                    if not kk:
                        continue
                    if c in PARALLEL_SAFE and args.jobs > 1:
                        results = Parallel(n_jobs=args.jobs)(
                            delayed(evaluate)(c, Z_tr[:, :k], y[tr], Z_va[:, :k], y[va], class_ids, args.seed)
                            for k in kk)
                    else:
                        results = [evaluate(c, Z_tr[:, :k], y[tr], Z_va[:, :k], y[va], class_ids, args.seed)
                                   for k in kk]
                    for k, res in zip(kk, results):
                        row = {f: "" for f in fields}
                        row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   run_id=run_id, scaler=sname, reduction=red, k=k, classifier=c, fold=fi,
                                   n_train=len(tr), n_valid=len(va), reducer_seconds=reducer_seconds, **res)
                        rows.append(row)
                append_rows(fold_csv, rows, fields)
                failed = sum(r["status"] == "failed" for r in rows)
                best = max((r for r in rows if r["status"] == "success"),
                           key=lambda r: r["f1_macro"], default=None)
                log(f"fold {fi}/{len(folds)} {sname:10} {red:9} {len(rows):3} fits, {failed} failed, "
                    f"reducer {reducer_seconds}s, block {time.perf_counter() - block_start:.0f}s"
                    + (f", best {best['classifier']} k={best['k']} macro-F1 {best['f1_macro']:.4f}" if best else "")
                    + f" | elapsed {time.perf_counter() - started:.0f}s")

    elapsed = time.perf_counter() - started
    summary = summarize(fold_csv, len(folds), data.class_names, out / "grid_summary.csv")
    cols = ["scaler", "reduction", "k", "classifier", "f1_macro_mean", "f1_macro_sd",
            "balanced_accuracy_mean", "accuracy_mean"]
    log(f"END elapsed {elapsed:.0f}s | complete configurations {len(summary)}")
    print("\nTop 10 configurations by macro-F1:")
    print(summary[cols].head(10).round(4).to_string(index=False))
    print("\nBest configuration per classifier:")
    print(summary.groupby("classifier", sort=False).head(1)[cols].round(4).to_string(index=False))
    print("\nBest configuration per reduction:")
    print(summary.groupby("reduction", sort=False).head(1)[cols].round(4).to_string(index=False))
    if args.smoke:
        full = 3 * (36 + 36 + 5) * 4 * 4
        log(f"Timing check: {n_cfg} fits in {elapsed:.0f}s. Estimated full-grid time: "
            f"{elapsed * full / n_cfg / 3600:.1f} h (upper bound; DT and SVM parallelise better over 36 k values)")


if __name__ == "__main__":
    main()
