"""Time each reducer and classifier from the professor's pipeline on samples.

Read-only timing check: no results are written. Run from the repository root:
    python scripts\\time_pcap_models.py
Expects data\\pcap.csv. Agent and Tsunami are excluded (too few hashes).
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
N_COMPONENTS = 10
SIZES = (5_000, 20_000, 50_000)
KPCA_MAX_ROWS = 5_000      # kernel matrix is n x n: 5k rows ~ 0.2 GB
SVM_MAX_ROWS = 20_000      # RBF-SVM training grows ~ n^2


def main():
    df = pd.read_csv("data/pcap.csv")
    df.columns = df.columns.str.strip()
    df = df[~df["MalwareFamily"].isin(["Agent", "Tsunami"])]
    X = (df.drop(columns=["Hash", "Arch", "MalwareFamily"])
           .apply(pd.to_numeric, errors="raise")
           .replace([np.inf, -np.inf], np.nan)
           .to_numpy(dtype=float))                         # Shape [N, 39]
    y = LabelEncoder().fit_transform(df["MalwareFamily"])  # Shape [N]
    print(f"rows {len(y):,} | features {X.shape[1]} | CPU cores {os.cpu_count()}")

    rng = np.random.default_rng(SEED)
    for n in SIZES:
        idx = rng.choice(len(y), size=n, replace=False)
        n_train = int(0.8 * n)
        tr, te = idx[:n_train], idx[n_train:]

        imputer = SimpleImputer(strategy="median").fit(X[tr])
        scaler = StandardScaler().fit(imputer.transform(X[tr]))
        Xtr = scaler.transform(imputer.transform(X[tr]))   # Shape [0.8n, 39]
        Xte = scaler.transform(imputer.transform(X[te]))   # Shape [0.2n, 39]

        times = {}
        t = time.perf_counter()
        pca = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)  # Shape [*, 10]
        times["PCA"] = time.perf_counter() - t

        if n <= KPCA_MAX_ROWS:
            t = time.perf_counter()
            KernelPCA(n_components=N_COMPONENTS, kernel="rbf", gamma=1.0 / X.shape[1],
                      eigen_solver="arpack", random_state=SEED).fit(Xtr).transform(Xte)
            times["KPCA"] = time.perf_counter() - t

        models = {
            "RF": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
            "DT": DecisionTreeClassifier(random_state=SEED),
            "XGB": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 tree_method="hist", random_state=SEED, n_jobs=-1),
        }
        if n <= SVM_MAX_ROWS:
            models["SVM"] = SVC(kernel="rbf", C=1.0, gamma="scale")

        for name, model in models.items():
            t = time.perf_counter()
            model.fit(Ztr, y[tr])
            model.predict(Zte)
            times[name] = time.perf_counter() - t

        print(f"n={n:>6,}  " + "  ".join(f"{k} {v:.2f}s" for k, v in times.items()), flush=True)


if __name__ == "__main__":
    main()
