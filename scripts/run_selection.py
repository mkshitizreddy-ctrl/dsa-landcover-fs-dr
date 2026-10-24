
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mi import relevance_scores
from graph import build_corr_graph, components
from selection import select_features, select_per_component, select_naive
from sorting import merge_sort


# 1. Load the official Statlog Landsat train/test split.
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

# The test split is loaded only to report its size.
# Its features and labels are never used for feature selection.
print(f"Training samples: {len(ytr)}")
print(f"Held-out test samples: {len(te)}")


# 2. Standardize using training statistics only.
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0
Xtr = (Xtr - mu) / sd


# 3. Calculate relevance and correlation from training data only.
scores = relevance_scores(Xtr, ytr)

C = np.abs(np.corrcoef(Xtr.T))
C = np.triu(C) + np.triu(C, 1).T

top = merge_sort(
    list(range(Xtr.shape[1])),
    key=lambda j: scores[j],
    reverse=True,
)
print("Top 5 features by MI (zero-based indices):", top[:5])


# 4. Compare feature-selection methods at several thresholds.
for threshold in (0.7, 0.8, 0.9, 0.95):
    graph = build_corr_graph(C, threshold)
    _, component_count = components(graph)

    selected = select_features(scores, graph)
    per_component = select_per_component(scores, graph)
    naive = select_naive(scores, C, threshold)

    print(
        f"t={threshold}: "
        f"components={component_count}, "
        f"heap+BFS={len(selected)}, "
        f"per-component={len(per_component)}, "
        f"same as naive={sorted(selected) == sorted(naive)}"
    )
