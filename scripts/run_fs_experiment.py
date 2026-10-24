
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# Resolve paths relative to the project, not the terminal's current directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mi import relevance_scores
from graph import build_corr_graph
from selection import select_features
from knn import KNN


# 1. Load the official train/test split from local files when available.
BASE_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/statlog/satimage/"
)

os.makedirs(DATA_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "sat.trn")
test_path = os.path.join(DATA_DIR, "sat.tst")

for filename, path in (
    ("sat.trn", train_path),
    ("sat.tst", test_path),
):
    if os.path.isfile(path):
        print(f"Using local dataset: {path}")
    else:
        url = BASE_URL + filename
        print(f"Local file missing; downloading {filename} from UCI...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved dataset to: {path}")

tr = pd.read_csv(train_path, sep=r"\s+", header=None)
te = pd.read_csv(test_path, sep=r"\s+", header=None)

Xtr = tr.iloc[:, :36].to_numpy(dtype=float)
ytr = tr.iloc[:, 36].to_numpy()
Xte = te.iloc[:, :36].to_numpy(dtype=float)
yte = te.iloc[:, 36].to_numpy()

print(f"Training samples: {len(ytr)}")
print(f"Test samples: {len(yte)}")


# 2. Standardize using training statistics only.
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)

# Avoid division by zero if a feature has no variation.
sd[sd == 0] = 1.0

Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd


# 3. Calculate relevance and correlations using training data only.
scores = relevance_scores(Xtr, ytr)
C = np.abs(np.corrcoef(Xtr.T))
C = np.triu(C) + np.triu(C, 1).T


def run(name, cols):
    # 4. Fit kNN on the selected training features and evaluate on test data.
    t0 = time.time()
    model = KNN(5).fit(Xtr[:, cols], ytr)
    t1 = time.time()

    pred = model.predict(Xte[:, cols])
    t2 = time.time()

    return {
        "setting": name,
        "features": len(cols),
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "build_s": round(t1 - t0, 2),
        "query_s": round(t2 - t1, 2),
    }


# 5. Evaluate raw features and feature selection at each threshold.
rows = [run("raw (all 36)", list(range(36)))]

for threshold in (0.7, 0.8, 0.9, 0.95):
    graph = build_corr_graph(C, threshold)
    selected = sorted(select_features(scores, graph))
    rows.append(run(f"heap+BFS t={threshold}", selected))


# 6. Compare with scikit-learn kNN on all raw features.
reference = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
reference_predictions = reference.predict(Xte)

print(
    "sklearn kNN on raw features: accuracy",
    round(accuracy_score(yte, reference_predictions), 4),
)

results = pd.DataFrame(rows)
print(results.to_string(index=False))

os.makedirs(RESULTS_DIR, exist_ok=True)
output_path = os.path.join(RESULTS_DIR, "fs_experiment.csv")
results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
