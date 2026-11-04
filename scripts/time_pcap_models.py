import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pcap_data import DATA_PATH, load_pcap
from pcap_grid import make_classifier

SIZES = (5_000, 20_000, 50_000)
N_COMPONENTS = 10
KPCA_MAX_ROWS = 5_000
SVM_MAX_ROWS = 20_000


def seconds(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def time_one_size(X, y, n, rng, seed):
    idx = rng.choice(len(y), size=n, replace=False)
    train, test = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    imputer = SimpleImputer(strategy="median").fit(X[train])
    scaler = StandardScaler().fit(imputer.transform(X[train]))
    X_train, X_test = scaler.transform(imputer.transform(X[train])), scaler.transform(imputer.transform(X[test]))

    row = {"rows": n}
    pca, row["PCA"] = seconds(lambda: PCA(n_components=N_COMPONENTS, svd_solver="full").fit(X_train))
    Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    if n <= KPCA_MAX_ROWS:
        _, row["KPCA"] = seconds(lambda: KernelPCA(n_components=N_COMPONENTS, kernel="rbf",
                                                   gamma=1.0 / X.shape[1], eigen_solver="arpack",
                                                   random_state=seed).fit(X_train).transform(X_test))
    models = ["RF", "DT", "XGB"] + (["SVM"] if n <= SVM_MAX_ROWS else [])
    for name in models:
        _, row[name] = seconds(lambda: make_classifier(name, seed).fit(Z_train, y[train]).predict(Z_test))
    return row


def time_pcap_models(data_path=DATA_PATH, sizes=SIZES, seed=42, show=print):
    data = load_pcap(data_path)
    X = data.X.to_numpy(dtype=float)
    show(f"Rows {len(data.y):,} | features {X.shape[1]} | CPU cores {os.cpu_count()}")
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        rows.append(time_one_size(X, data.y, n, rng, seed))
        show("  ".join(f"{k} {v:.2f}s" if k != "rows" else f"n={v:,}" for k, v in rows[-1].items()))
    return pd.DataFrame(rows)
