import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from mi import relevance_scores
from graph import build_corr_graph, components
from selection import select_features, select_per_component, select_naive
from sorting import merge_sort

# 1. load Statlog Landsat
base = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/"
tr = pd.read_csv(base + "sat.trn", sep=r"\s+", header=None)
te = pd.read_csv(base + "sat.tst", sep=r"\s+", header=None)
df = pd.concat([tr, te], ignore_index=True)
X = df.iloc[:, :36].to_numpy(dtype=float)
y = df.iloc[:, 36].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 2. relevance and correlation
scores = relevance_scores(X, y)
C = np.abs(np.corrcoef(X.T))
C = np.triu(C) + np.triu(C, 1).T
top = merge_sort(list(range(36)), key=lambda j: scores[j], reverse=True)
print("top 5 features by MI:", top[:5])

# 3. selection at several thresholds
for t in (0.7, 0.8, 0.9, 0.95):
    g = build_corr_graph(C, t)
    _, k = components(g)
    S = select_features(scores, g)
    P = select_per_component(scores, g)
    N = select_naive(scores, C, t)
    print(f"t={t}: components={k}, heap+BFS={len(S)}, per-component={len(P)}, same as naive={sorted(S) == sorted(N)}")