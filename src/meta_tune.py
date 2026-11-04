import time
from dataclasses import dataclass, field
from math import ceil

import numpy as np
from mealpy import FloatVar
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from graph import build_corr_graph
from meta_fs import ALGORITHM_FAMILY, build_optimizer
from mi import relevance_scores
from pca import PCA
from selection import select_features

MODES = ("threshold", "components", "joint")
SEARCH_DIMENSIONS = {"threshold": 1, "components": 1, "joint": 2}


def decode_k(k_frac, m):
    return int(min(m, max(1, ceil(float(k_frac) * m - 1e-9))))


class PipelineObjective:
    def __init__(self, X_train, y_train, X_valid, y_valid, mode="joint",
                 alpha=0.9, k_neighbors=5, pca_backend="custom", mi_bins=10):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if pca_backend not in ("custom", "svd"):
            raise ValueError("pca_backend must be 'custom' or 'svd'")
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.D = X_train.shape[1]
        self.mode, self.alpha = mode, alpha
        self.k_neighbors, self.pca_backend = k_neighbors, pca_backend

        self.C = np.nan_to_num(np.abs(np.corrcoef(X_train, rowvar=False)), nan=0.0)
        self.scores = relevance_scores(X_train, y_train, bins=mi_bins)
        self.breakpoints = np.unique(self.C[np.triu_indices(self.D, k=1)])

        self.subset_cache, self.pca_cache, self.score_cache = {}, {}, {}
        self.calls = 0
        self.best = None

    def subset_for(self, t):
        key = int(np.searchsorted(self.breakpoints, t, side="right"))
        if key not in self.subset_cache:
            graph = build_corr_graph(self.C, t)
            self.subset_cache[key] = tuple(sorted(select_features(self.scores, graph)))
        return self.subset_cache[key]

    def pca_projection(self, S):
        if S not in self.pca_cache:
            Xs = self.X_train[:, list(S)]
            if self.pca_backend == "custom":
                model = PCA(max_components=len(S)).fit(Xs)
                self.pca_cache[S] = (model.mean, model.std, model.components)
            else:
                mean, std = Xs.mean(axis=0), Xs.std(axis=0)
                std[std == 0] = 1.0
                _, _, Vt = np.linalg.svd((Xs - mean) / std, full_matrices=False)
                self.pca_cache[S] = (mean, std, Vt)
        return self.pca_cache[S]

    def evaluate(self, S, k, use_pca):
        key = (S, k, use_pca)
        if key not in self.score_cache:
            cols = list(S)
            if use_pca:
                mean, std, W = self.pca_projection(S)
                Z_train = ((self.X_train[:, cols] - mean) / std) @ W[:k].T
                Z_valid = ((self.X_valid[:, cols] - mean) / std) @ W[:k].T
            else:
                Z_train, Z_valid = self.X_train[:, cols], self.X_valid[:, cols]
            clf = KNeighborsClassifier(n_neighbors=self.k_neighbors).fit(Z_train, self.y_train)
            f1 = f1_score(self.y_valid, clf.predict(Z_valid), average="macro", zero_division=0)
            fitness = self.alpha * (1.0 - f1) + (1.0 - self.alpha) * k / self.D
            self.score_cache[key] = (float(fitness), float(f1))
        return self.score_cache[key]

    def decode(self, position):
        position = np.atleast_1d(np.asarray(position, dtype=float))
        if self.mode == "threshold":
            t = float(position[0])
            S = self.subset_for(t)
            return t, S, len(S), False
        if self.mode == "components":
            S = tuple(range(self.D))
            return None, S, decode_k(position[0], self.D), True
        t = float(position[0])
        S = self.subset_for(t)
        return t, S, decode_k(position[1], len(S)), True

    def __call__(self, position):
        self.calls += 1
        t, S, k, use_pca = self.decode(position)
        fitness, f1 = self.evaluate(S, k, use_pca)
        if self.best is None or fitness < self.best[0]:
            self.best = (fitness, f1, t, k, S)
        return fitness

    def bounds(self):
        n = SEARCH_DIMENSIONS[self.mode]
        return FloatVar(lb=[0.0] * n, ub=[1.0] * n)


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


def make_result(name, objective, runtime, convergence=()):
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
    objective = PipelineObjective(X_train, y_train, X_valid, y_valid, mode=mode,
                                  alpha=alpha, k_neighbors=k_neighbors, pca_backend=pca_backend)
    if max_evaluations < max(10, population):
        raise ValueError("max_evaluations must be >= max(10, population)")
    problem = {"obj_func": objective, "bounds": objective.bounds(), "minmax": "min", "log_to": None}
    optimizer = build_optimizer(name, 10_000, population)
    start = time.perf_counter()
    optimizer.solve(problem, seed=seed, termination={"max_fe": int(max_evaluations)})
    runtime = time.perf_counter() - start
    curve = [float(v) for v in optimizer.history.list_global_best_fit]
    return make_result(name, objective, runtime, curve)


def grid_search(X_train, y_train, X_valid, y_valid, mode="joint",
                alpha=0.9, k_neighbors=5, pca_backend="custom"):
    objective = PipelineObjective(X_train, y_train, X_valid, y_valid, mode=mode,
                                  alpha=alpha, k_neighbors=k_neighbors, pca_backend=pca_backend)
    start = time.perf_counter()
    if mode == "components":
        for k in range(1, objective.D + 1):
            objective([k / objective.D])
    else:
        seen = set()
        for t in [0.0] + [float(v) for v in objective.breakpoints]:
            S = objective.subset_for(t)
            if S in seen:
                continue
            seen.add(S)
            if mode == "threshold":
                objective([t])
            else:
                for k in range(1, len(S) + 1):
                    objective([t, k / len(S)])
    return make_result("GRID", objective, time.perf_counter() - start)
