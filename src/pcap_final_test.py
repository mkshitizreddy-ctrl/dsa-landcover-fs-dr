import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_grid import CLASSIFIERS, CONFIG, GRID_DIR, fit_reducer, make_classifier, make_scaler, transform

ROOT = Path(__file__).resolve().parents[1]
GRID_SUMMARY_PATH = GRID_DIR / "grid_summary.csv"
FINAL_DIR = ROOT / "results" / "pcap_final_v1"
EXPECTED_CONFIGS = 924
CLASSIFIER_TAG = {"RF": "C1", "XGB": "C2", "DT": "C3", "SVM": "C4"}


def select_configs(grid):
    picks = {}
    for metric, label in (("accuracy_mean", "accuracy"), ("f1_macro_mean", "macro-F1")):
        ranked = grid.sort_values(metric, ascending=False, kind="stable")
        top = ranked.iloc[0]
        picks.setdefault(tuple(top[CONFIG]), []).append(f"best overall by {label}")
        for clf in CLASSIFIERS:
            row = ranked[ranked.classifier == clf].iloc[0]
            picks.setdefault(tuple(row[CONFIG]), []).append(f"{CLASSIFIER_TAG[clf]} ({clf}) by {label}")
    selected = []
    for cfg, reasons in picks.items():
        row = grid[(grid.scaler == cfg[0]) & (grid.reduction == cfg[1]) &
                   (grid.k == cfg[2]) & (grid.classifier == cfg[3])].iloc[0]
        selected.append({"scaler": cfg[0], "reduction": cfg[1], "k": int(cfg[2]), "classifier": cfg[3],
                         "chosen_as": "; ".join(reasons), "cv_accuracy": row.accuracy_mean,
                         "cv_f1_macro": row.f1_macro_mean, "cv_balanced_accuracy": row.balanced_accuracy_mean})
    return selected


def fit_config(cfg, X_train, y_train, seed=42, kpca_max_train=8000, svm_max_rows=60000):
    if cfg["classifier"] == "SVM" and svm_max_rows and len(y_train) > svm_max_rows:
        rows, _ = train_test_split(np.arange(len(y_train)), train_size=svm_max_rows,
                                   stratify=y_train, random_state=seed)
        X_train, y_train = X_train[rows], y_train[rows]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train)
    imputed = imputer.transform(X_train)
    scaler = make_scaler(cfg["scaler"]).fit(imputed)
    scaled = scaler.transform(imputed)
    reducer = fit_reducer(cfg["reduction"], scaled, y_train, cfg["k"], seed=seed, kpca_max_train=kpca_max_train)
    model = make_classifier(cfg["classifier"], seed).fit(transform(reducer, scaled), y_train)
    pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("reducer", reducer), ("model", model)])
    return pipe, len(y_train)


def predict_chunked(pipe, X, chunk=4000):
    return np.concatenate([pipe.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def config_tag(cfg):
    return f"{cfg['scaler']}_{cfg['reduction']}_k{cfg['k']}_{cfg['classifier']}"


def save_test_outputs(out_dir, tag, y_test, pred, class_names):
    labels = np.arange(len(class_names))
    report = classification_report(y_test, pred, labels=labels, target_names=class_names, zero_division=0)
    (out_dir / f"test_report_{tag}.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=class_names,
                 columns=class_names).to_csv(out_dir / f"test_confusion_{tag}.csv")
    return report


def evaluate_on_test(cfg, X_dev, y_dev, X_test, y_test, class_names, out_dir, seed, kpca_max_train, svm_max_rows):
    start = time.perf_counter()
    pipe, n_train = fit_config(cfg, X_dev, y_dev, seed=seed, kpca_max_train=kpca_max_train,
                               svm_max_rows=svm_max_rows)
    fit_seconds = time.perf_counter() - start
    pred = predict_chunked(pipe, X_test)
    report = save_test_outputs(out_dir, config_tag(cfg), y_test, pred, class_names)
    row = {**cfg, "train_rows": n_train,
           "test_accuracy": accuracy_score(y_test, pred),
           "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
           "test_f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
           "fit_seconds": round(fit_seconds, 1),
           "total_seconds": round(time.perf_counter() - start, 1)}
    return row, pipe, report


def save_selected_model(rows, pipes, class_names, features, out_dir):
    selected = next(r for r in rows if "best overall by accuracy" in r["chosen_as"])
    tag = config_tag(selected)
    pd.DataFrame([selected]).to_csv(out_dir / "best_configuration.csv", index=False)
    joblib.dump({"pipeline": pipes[tag], "class_names": class_names, "feature_names": features,
                 "configuration": {c: selected[c] for c in CONFIG}}, out_dir / "best_model.joblib", compress=3)
    return tag


def run_final_test(data_path=DATA_PATH, splits_path=SPLITS_PATH, grid_path=GRID_SUMMARY_PATH, out_dir=FINAL_DIR,
                   svm_max_rows=60000, kpca_max_train=8000, seed=42, show=print):
    out_dir = Path(out_dir)
    if (out_dir / "test_results.csv").exists():
        raise FileExistsError(f"{out_dir / 'test_results.csv'} exists: the test set has already been used. "
                              "Choose a new output folder only if you mean to evaluate again.")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    X, y = data.X[splits.features].to_numpy(dtype=float), data.y
    X_dev, y_dev = X[splits.dev_idx], y[splits.dev_idx]
    X_test, y_test = X[splits.test_idx], y[splits.test_idx]
    show(f"Development rows {len(y_dev):,} | test rows {len(y_test):,} | features {X.shape[1]}")

    grid = pd.read_csv(grid_path)
    if len(grid) != EXPECTED_CONFIGS:
        raise ValueError(f"Grid summary has {len(grid)} configurations, expected {EXPECTED_CONFIGS}")
    configs = select_configs(grid)
    show(f"Configurations to evaluate: {len(configs)}")

    rows, pipes = [], {}
    for i, cfg in enumerate(configs, start=1):
        row, pipe, _ = evaluate_on_test(cfg, X_dev, y_dev, X_test, y_test, data.class_names, out_dir,
                                             seed, kpca_max_train, svm_max_rows)
        rows.append(row)
        pipes[config_tag(cfg)] = pipe
        pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)
        show(f"[{i}/{len(configs)}] {config_tag(cfg)}: test accuracy {row['test_accuracy']:.4f}, "
             f"macro-F1 {row['test_f1_macro']:.4f}, balanced {row['test_balanced_accuracy']:.4f}")

    tag = save_selected_model(rows, pipes, data.class_names, splits.features, out_dir)
    show(f"Selected model (highest CV accuracy): {tag}. Saved as best_model.joblib")
    return pd.DataFrame(rows)
