import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from graph import build_corr_graph
from selection import select_features, select_naive

def timed(f, *a):
    # 1. wall-clock time of one call
    t0 = time.perf_counter()
    out = f(*a)
    return out, time.perf_counter() - t0

rng = np.random.default_rng(0)
m, t = 500, 0.7
rows = []
for d in (50, 100, 200, 400, 800, 1600):
    # 2. synthetic data: blocks of 4 correlated features (corr about 0.8)
    Z = rng.normal(size=(m, d // 4 + 1))
    X = np.column_stack([Z[:, j // 4] + 0.5 * rng.normal(size=m) for j in range(d)])
    scores = list(rng.random(d))

    # 3. correlation matrix, O(d^2 m)
    C, t_corr = timed(lambda: np.abs(np.corrcoef(X.T)))
    C = np.triu(C) + np.triu(C, 1).T

    # 4. graph build O(d^2), heap+BFS selection, naive baseline
    g, t_graph = timed(build_corr_graph, C, t)
    S, t_sel = timed(select_features, scores, g)
    N, t_naive = timed(select_naive, scores, C, t)
    assert sorted(S) == sorted(N)

    rows.append({"d": d, "edges": sum(len(a) for a in g.adj) // 2, "kept": len(S),
                 "corr_s": t_corr, "graph_s": t_graph, "heap_bfs_select_s": t_sel,
                 "heap_bfs_total_s": t_graph + t_sel, "naive_s": t_naive})

df = pd.DataFrame(rows)
print(df.round(5).to_string(index=False))

# 5. empirical exponent: slope of log(time) vs log(d), d >= 200
print("\nGrowth exponent (time ~ d^k), fitted on d >= 200:")
for c in ("corr_s", "graph_s", "heap_bfs_select_s", "heap_bfs_total_s", "naive_s"):
    k = np.polyfit(np.log(df.d[2:]), np.log(df[c][2:]), 1)[0]
    print(f"  {c}: k = {k:.2f}")

# 6. log-log plot
os.makedirs("results", exist_ok=True)
for c in ("corr_s", "graph_s", "heap_bfs_select_s", "naive_s"):
    plt.loglog(df.d, df[c], marker="o", label=c)
plt.xlabel("number of features d")
plt.ylabel("time (s)")
plt.title("Runtime vs number of features")
plt.legend()
plt.savefig("results/complexity.png", dpi=200, bbox_inches="tight")
df.to_csv("results/complexity.csv", index=False)