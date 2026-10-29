"""Metaheuristic tuning of the DSA pipeline's own knobs.

Instead of searching feature masks directly (meta_fs.py), the search picks
the two settings our pipeline otherwise hand-picks:

    t : correlation threshold for heap + BFS selection (graph.py, selection.py)
    k : number of principal components kept by PCA (pca.py)

Pipeline for a candidate x:
    S(t) = select_features(MI scores, corr-graph(|C| > t))     # our heap + BFS
    Z    = PCA_k(X[:, S(t)])                                   # our PCA, k <= |S(t)|
    f(x) = alpha * (1 - macroF1_valid(kNN on Z)) + (1 - alpha) * k / D

Modes:
    "threshold"  : x = t,          no PCA, k = |S(t)|
    "components" : x = k_frac,     PCA on all D features
    "joint"      : x = (t, k_frac)

k is encoded as a fraction k_frac in (0, 1] and decoded as ceil(k_frac * m),
where m = |S(t)| (or D), so every candidate is feasible (k <= m) without
penalty terms.

Exactness note: S(t) only changes when t crosses one of the observed
off-diagonal |C_ij| values, so each knob has finitely many distinct outcomes.
grid_search() enumerates all of them and gives the true optimum, which is the
reference the metaheuristics are scored against.

As in meta_fs.py, only the inner train/validation split passed in is used.
"""

from dataclasses import dataclass, field
from math import ceil
import time

import numpy as np
from mealpy import FloatVar
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from graph import build_corr_graph
from mi import relevance_scores
from pca import PCA
from selection import select_features
from meta_fs import ALGORITHM_FAMILY, build_optimizer

MODES = ("threshold", "components", "joint")


def decode_k(k_frac, m):
    """Fraction in [0, 1] -> integer component count in [1, m]."""
    # The 1e-9 guard stops k/m * m = k + eps from rounding up to k + 1.
    return int(min(m, max(1, ceil(float(k_frac) * m - 1e-9))))


class PipelineObjective:
    """Fitness over (t, k) with caches for subsets, PCA fits and scores.

    Caches (all hash maps):
      subset_cache : breakpoint index -> S(t)      (graph + heap/BFS run once)
      pca_cache    : S -> fitted PCA with all |S| components
      score_cache  : (S, k) -> (fitness, macro_f1)
    The custom PCA is fitted once per distinct subset with every component,
    and a candidate k just slices the first k projections.
    """

    def __init__(self, X_train, y_train, X_valid, y_valid, mode="joint",
                 alpha=0.9, k_neighbors=5, pca_backend="custom", mi_bins=10):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if pca_backend not in ("custom", "svd"):
            raise ValueError("pca_backend must be 'custom' or 'svd'")
        # X_train: Shape [N_train, D];  X_valid: Shape [N_valid, D]
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.D = X_train.shape[1]
        self.mode, self.alpha = mode, alpha
        self.k_neighbors, self.pca_backend = k_neighbors, pca_backend

        # Fit-once statistics, from inner-training rows only.
        C = np.abs(np.corrcoef(X_train, rowvar=False))          # Shape [D, D]
        C = np.nan_to_num(C, nan=0.0)                           # constant columns
        self.C = C
        self.scores = relevance_scores(X_train, y_train, bins=mi_bins)  # len D
        off_diag = C[np.triu_indices(self.D, k=1)]
        self.breakpoints = np.unique(off_diag)                  # sorted |C_ij|

        self.subset_cache, self.pca_cache, self.score_cache = {}, {}, {}
        self.calls = 0
        self.best = None    # (fitness, f1, t, k, S)

    # ---- pipeline pieces -------------------------------------------------
    def subset_for(self, t):
        """S(t) via our graph + heap/BFS; cached by breakpoint index."""
        # Edges are |C_ij| > t, so S(t) depends only on how many breakpoints are <= t.
        key = int(np.searchsorted(self.breakpoints, t, side="right"))
        if key not in self.subset_cache:
            g = build_corr_graph(self.C, t)
            self.subset_cache[key] = tuple(sorted(select_features(self.scores, g)))
        return self.subset_cache[key]

    def _pca(self, S):
        if S not in self.pca_cache:
            Xs = self.X_train[:, list(S)]                       # Shape [N_train, |S|]
            if self.pca_backend == "custom":
                model = PCA(max_components=len(S)).fit(Xs)
                proj = (model.mean, model.std, model.components)   # components: [|S|, |S|]
            else:
                mean, std = Xs.mean(axis=0), Xs.std(axis=0)
                std[std == 0] = 1.0
                _, _, Vt = np.linalg.svd((Xs - mean) / std, full_matrices=False)
                proj = (mean, std, Vt)
            self.pca_cache[S] = proj
        return self.pca_cache[S]

    def evaluate(self, S, k, use_pca):
        """Fitness and macro-F1 for subset S reduced to k dims."""
        key = (S, k, use_pca)
        if key not in self.score_cache:
            cols = list(S)
            if use_pca:
                mean, std, W = self._pca(S)
                Ztr = ((self.X_train[:, cols] - mean) / std) @ W[:k].T   # [N_train, k]
                Zva = ((self.X_valid[:, cols] - mean) / std) @ W[:k].T   # [N_valid, k]
            else:
                Ztr, Zva = self.X_train[:, cols], self.X_valid[:, cols]
            clf = KNeighborsClassifier(n_neighbors=self.k_neighbors).fit(Ztr, self.y_train)
            f1 = f1_score(self.y_valid, clf.predict(Zva), average="macro", zero_division=0)
            fitness = self.alpha * (1.0 - f1) + (1.0 - self.alpha) * k / self.D
            self.score_cache[key] = (float(fitness), float(f1))
        return self.score_cache[key]

    def decode(self, position):
        """position -> (t, S, k, use_pca) for the current mode."""
        position = np.atleast_1d(np.asarray(position, dtype=float))
        if self.mode == "threshold":
            t = float(position[0]); S = self.subset_for(t)
            return t, S, len(S), False
        if self.mode == "components":
            S = tuple(range(self.D))
            return None, S, decode_k(position[0], self.D), True
        t = float(position[0]); S = self.subset_for(t)
        return t, S, decode_k(position[1], len(S)), True

    def __call__(self, position):
        self.calls += 1
        t, S, k, use_pca = self.decode(position)
        fitness, f1 = self.evaluate(S, k, use_pca)
        if self.best is None or fitness < self.best[0]:
            self.best = (fitness, f1, t, k, S)
        return fitness

    def bounds(self):
        dims = {"threshold": 1, "components": 1, "joint": 2}[self.mode]
        # t in [0, 1]; k_frac in [0, 1] (decode_k clamps to >= 1).
        return FloatVar(lb=[0.0] * dims, ub=[1.0] * dims)


@dataclass
class TuneResult:
    algorithm: str
    family: str
    mode: str
    threshold: float | None
    n_components: int
    selected_indices: tuple
    best_fitness: float
    best_inner_f1: float
    calls: int
    unique_evaluations: int
    runtime_seconds: float
    convergence: list = field(default_factory=list)


def _result(name, objective, runtime, convergence=()):
    fitness, f1, t, k, S = objective.best
    return TuneResult(
        algorithm=name, family=ALGORITHM_FAMILY.get(name, "exhaustive"),
        mode=objective.mode, threshold=t, n_components=k, selected_indices=S,
        best_fitness=fitness, best_inner_f1=f1, calls=objective.calls,
        unique_evaluations=len(objective.score_cache),
        runtime_seconds=round(runtime, 4), convergence=list(convergence),
    )


def run_tuning(name, X_train, y_train, X_valid, y_valid, mode="joint",
               population=10, seed=42, max_evaluations=100, alpha=0.9,
               k_neighbors=5, pca_backend="custom"):
    """One metaheuristic, one inner split, equal evaluation budget."""
    objective = PipelineObjective(X_train, y_train, X_valid, y_valid, mode=mode,
                                  alpha=alpha, k_neighbors=k_neighbors,
                                  pca_backend=pca_backend)
    if max_evaluations < max(10, population):
        raise ValueError("max_evaluations must be >= max(10, population)")
    problem = {"obj_func": objective, "bounds": objective.bounds(),
               "minmax": "min", "log_to": None}
    optimizer = build_optimizer(name, 10_000, population)
    start = time.perf_counter()
    optimizer.solve(problem, seed=seed, termination={"max_fe": int(max_evaluations)})
    runtime = time.perf_counter() - start
    curve = [float(v) for v in optimizer.history.list_global_best_fit]
    return _result(name, objective, runtime, curve)


def grid_search(X_train, y_train, X_valid, y_valid, mode="joint",
                alpha=0.9, k_neighbors=5, pca_backend="custom"):
    """Exhaustive search over every distinct (S(t), k): the true optimum."""
    objective = PipelineObjective(X_train, y_train, X_valid, y_valid, mode=mode,
                                  alpha=alpha, k_neighbors=k_neighbors,
                                  pca_backend=pca_backend)
    start = time.perf_counter()
    if mode == "components":
        for k in range(1, objective.D + 1):
            objective([k / objective.D])
    else:
        # One representative t per distinct subset: below every breakpoint,
        # between consecutive breakpoints, and at/above the largest one.
        bp = objective.breakpoints
        reps = [0.0] + [float(v) for v in bp]   # t == bp[i] keeps edges > bp[i]
        seen = set()
        for t in reps:
            S = objective.subset_for(t)
            if S in seen:
                continue
            seen.add(S)
            if mode == "threshold":
                objective([t])
            else:
                for k in range(1, len(S) + 1):
                    objective([t, k / len(S)])
    runtime = time.perf_counter() - start
    return _result("GRID", objective, runtime)
