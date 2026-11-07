import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from pcap_data import DATA_PATH, SPLITS_PATH, load_pcap, load_splits
from pcap_feature_selection import FS_DIR, METHODS, feature_statistics, heap_bfs_subsets, mi_top_subsets
from pcap_final_test import FINAL_DIR, predict_chunked, save_test_outputs
from pcap_fs_search import MASK_DIR
from pcap_grid import CLASSIFIERS, make_classifier, make_scaler

ROOT = Path(__file__).resolve().parents[1]
FS_SUMMARY_PATH = FS_DIR / "fs_summary.csv"
MASK_RUNS_PATH = MASK_DIR / "runs.csv"
FS_FINAL_DIR = ROOT / "results" / "pcap_fs_final_v1"
GRID_TEST_PATH = FINAL_DIR / "test_results.csv"


class SelectedModel:

    def __init__(self, imputer, scaler, cols, model):
        self.imputer, self.scaler, self.cols, self.model = imputer, scaler, cols, model

    def predict(self, X):
        return self.model.predict(self.scaler.transform(self.imputer.transform(X))[:, self.cols])


def setting_key(row):
    return (row.scaler, row.method, float(row.param), row.classifier)


def select_settings(fs_summary, mask_runs):
    picks = {}
    for method in METHODS:
        ranked = fs_summary[fs_summary.method == method].sort_values("f1_macro_mean", ascending=False, kind="stable")
        picks.setdefault(setting_key(ranked.iloc[0]), []).append(f"best {method} overall")
        for clf in CLASSIFIERS:
            part = ranked[ranked.classifier == clf]
            if len(part):
                picks.setdefault(setting_key(part.iloc[0]), []).append(f"best {method} for {clf}")
    settings = []
    for (scaler, method, param, clf), reasons in picks.items():
        row = fs_summary[(fs_summary.scaler == scaler) & (fs_summary.method == method) &
                         (fs_summary.param == param) & (fs_summary.classifier == clf)].iloc[0]
        settings.append({"scaler": scaler, "method": method, "param": param, "classifier": clf, "features": "",
                         "chosen_as": "; ".join(reasons), "cv_f1_macro": row.f1_macro_mean,
                         "cv_accuracy": row.accuracy_mean})
    best = mask_runs.sort_values("cv_f1_macro", ascending=False, kind="stable").iloc[0]
    settings.append({"scaler": best.scaler, "method": "Mask", "param": f"{best.algorithm} seed {best.seed}",
                     "classifier": best.classifier, "features": best.features,
                     "chosen_as": "best metaheuristic mask", "cv_f1_macro": best.cv_f1_macro, "cv_accuracy": np.nan})
    return settings


def setting_tag(setting):
    if setting["method"] == "Mask":
        param = setting["param"].replace(" ", "_")
    elif setting["method"] == "HeapBFS":
        param = f"t{setting['param']:.2f}"
    else:
        param = f"k{int(setting['param'])}"
    return f"{setting['scaler']}_{setting['method']}_{param}_{setting['classifier']}"


def training_rows(setting, y_dev, svm_max_rows, seed):
    rows = np.arange(len(y_dev))
    if setting["classifier"] == "SVM" and svm_max_rows and len(rows) > svm_max_rows:
        rows, _ = train_test_split(rows, train_size=svm_max_rows, stratify=y_dev, random_state=seed)
    return np.sort(rows)


def choose_columns(setting, Z_train, y_train, feature_names, cache):
    if setting["method"] == "Mask":
        return [feature_names.index(name) for name in setting["features"].split(";")]
    key = (setting["scaler"], len(y_train))
    if key not in cache:
        cache[key] = feature_statistics(Z_train, y_train)
    scores, C = cache[key]
    if setting["method"] == "MI_top":
        k = int(setting["param"])
        return list(mi_top_subsets(scores, [k])[k])
    t = float(setting["param"])
    return list(heap_bfs_subsets(scores, C, [t])[t])


def fit_setting(setting, X_dev, y_dev, feature_names, cache, seed=42, svm_max_rows=60000):
    rows = training_rows(setting, y_dev, svm_max_rows, seed)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_dev[rows])
    A = imputer.transform(X_dev[rows])
    scaler = make_scaler(setting["scaler"]).fit(A)
    Z = scaler.transform(A)
    cols = choose_columns(setting, Z, y_dev[rows], feature_names, cache)
    model = make_classifier(setting["classifier"], seed).fit(Z[:, cols], y_dev[rows])
    return SelectedModel(imputer, scaler, cols, model), len(rows)


def compare_with_grid_test(results, grid_test_path=GRID_TEST_PATH):
    ours = results.assign(param=results.param.astype(str))
    grid = pd.read_csv(grid_test_path).rename(columns={"reduction": "method", "k": "param"})
    grid = grid.assign(param=grid.param.astype(str), n_features=grid.param)
    columns = ["scaler", "method", "param", "classifier", "n_features", "chosen_as", "cv_f1_macro",
               "test_f1_macro", "test_balanced_accuracy", "test_accuracy"]
    table = pd.concat([ours[columns], grid[columns]], ignore_index=True)
    return table.sort_values("test_f1_macro", ascending=False, kind="stable").reset_index(drop=True)


def run_fs_final_test(data_path=DATA_PATH, splits_path=SPLITS_PATH, fs_summary_path=FS_SUMMARY_PATH,
                      mask_runs_path=MASK_RUNS_PATH, out_dir=FS_FINAL_DIR, svm_max_rows=60000, seed=42, show=print):
    out_dir = Path(out_dir)
    if (out_dir / "test_results.csv").exists():
        raise FileExistsError(f"{out_dir / 'test_results.csv'} exists: these settings have already been tested. "
                              "Choose a new output folder only if you mean to evaluate again.")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_pcap(data_path)
    splits = load_splits(splits_path, n_rows=len(data.y))
    feature_names = list(splits.features)
    X, y = data.X[feature_names].to_numpy(dtype=float), data.y
    X_dev, y_dev = X[splits.dev_idx], y[splits.dev_idx]
    X_test, y_test = X[splits.test_idx], y[splits.test_idx]
    settings = select_settings(pd.read_csv(fs_summary_path), pd.read_csv(mask_runs_path))
    show(f"Development rows {len(y_dev):,} | test rows {len(y_test):,} | settings to evaluate: {len(settings)}")

    cache, rows = {}, []
    for i, setting in enumerate(settings, start=1):
        start = time.perf_counter()
        fitted, n_train = fit_setting(setting, X_dev, y_dev, feature_names, cache, seed, svm_max_rows)
        fit_seconds = time.perf_counter() - start
        pred = predict_chunked(fitted, X_test)
        save_test_outputs(out_dir, setting_tag(setting), y_test, pred, data.class_names)
        row = {**setting, "features": ";".join(feature_names[j] for j in fitted.cols), "n_features": len(fitted.cols),
               "train_rows": n_train, "test_accuracy": accuracy_score(y_test, pred),
               "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
               "test_f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
               "fit_seconds": round(fit_seconds, 1), "total_seconds": round(time.perf_counter() - start, 1)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)
        show(f"[{i}/{len(settings)}] {setting_tag(setting)} ({len(fitted.cols)} features): "
             f"test macro-F1 {row['test_f1_macro']:.4f}, accuracy {row['test_accuracy']:.4f}, "
             f"balanced {row['test_balanced_accuracy']:.4f} | {row['total_seconds']:.0f}s")

    results = pd.DataFrame(rows)
    if Path(GRID_TEST_PATH).exists():
        compare_with_grid_test(results).to_csv(out_dir / "comparison_with_grid_test.csv", index=False)
    return results
