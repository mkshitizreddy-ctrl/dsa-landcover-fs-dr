import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, "src")
from mi import relevance_scores
from graph import build_corr_graph
from selection import select_features
from knn import KNN

# 1. load the official train/test split
base = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/"
tr = pd.read_csv(base + "sat.trn", sep=r"\s+", header=None)
te = pd.read_csv(base + "sat.tst", sep=r"\s+", header=None)
Xtr, ytr = tr.iloc[:, :36].to_numpy(float), tr.iloc[:, 36].to_numpy()
Xte, yte = te.iloc[:, :36].to_numpy(float), te.iloc[:, 36].to_numpy()

# 2. standardize with training statistics only
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

# 3. relevance and correlation from the training set only
scores = relevance_scores(Xtr, ytr)
C = np.abs(np.corrcoef(Xtr.T))
C = np.triu(C) + np.triu(C, 1).T

def run(name, cols):
    # 4. fit our kNN on the chosen columns, time build and query
    t0 = time.time()
    model = KNN(5).fit(Xtr[:, cols], ytr)
    t1 = time.time()
    pred = model.predict(Xte[:, cols])
    t2 = time.time()
    return {"setting": name, "features": len(cols),
            "accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "build_s": round(t1 - t0, 2), "query_s": round(t2 - t1, 2)}

# 5. raw features, then selection at each threshold
rows = [run("raw (all 36)", list(range(36)))]
for t in (0.7, 0.8, 0.9, 0.95):
    g = build_corr_graph(C, t)
    S = sorted(select_features(scores, g))
    rows.append(run(f"heap+BFS t={t}", S))

# 6. reference: sklearn kNN on raw features
ref = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).predict(Xte)
print("sklearn kNN on raw features: accuracy", round(accuracy_score(yte, ref), 4))

res = pd.DataFrame(rows)
print(res.to_string(index=False))
os.makedirs("results", exist_ok=True)
res.to_csv("results/fs_experiment.csv", index=False)