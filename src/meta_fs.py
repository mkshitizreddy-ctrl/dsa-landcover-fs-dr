"""Wrapper feature selection with Mealpy metaheuristics.

Every algorithm searches the same continuous box [0, 1]^D and decodes a
position into a binary feature mask with a fixed threshold, so the encoding,
fitness and evaluation budget are identical across algorithms. Only the search
rule differs, which is what a fair comparative study needs.

Fitness (minimised), matching scripts/run_pso_nested_cv_v5.py by default:

    f(s) = alpha * (1 - macroF1_valid(s)) + (1 - alpha) * |s| / D

The fitness only ever sees the inner train/validation split passed in by the
caller. Outer folds and official test data must never reach this module.
"""

from dataclasses import dataclass, field
import time

import numpy as np
import mealpy
from mealpy import FloatVar
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


# name -> (mealpy module, class, extra constructor kwargs)
# Defaults are the Mealpy defaults. Note: Mealpy's "SSA" is the Sparrow
# Search Algorithm, not the Salp Swarm Algorithm.
ALGORITHMS = {
    "PSO": ("PSO", "OriginalPSO", {}),
    "GA": ("GA", "BaseGA", {}),
    "DE": ("DE", "OriginalDE", {}),
    "GWO": ("GWO", "OriginalGWO", {}),
    "WOA": ("WOA", "OriginalWOA", {}),
    "HHO": ("HHO", "OriginalHHO", {}),
    "SMA": ("SMA", "OriginalSMA", {}),
    "ABC": ("ABC", "OriginalABC", {}),
    "MFO": ("MFO", "OriginalMFO", {}),
    "SSA": ("SSA", "OriginalSSA", {}),
}

ALGORITHM_FAMILY = {
    "GA": "evolutionary", "DE": "evolutionary",
    "PSO": "swarm", "GWO": "swarm", "WOA": "swarm", "HHO": "swarm",
    "ABC": "swarm", "MFO": "swarm", "SSA": "swarm",
    "SMA": "bio",
}


# Mealpy 3.0.3 BaseGA hard-codes a 10% tournament (int(0.1 * pop)) in its
# mutation step and a 20% tournament for parents, so pop_size < 10 crashes.
# One shared minimum keeps every algorithm on the same budget.
MIN_POPULATION = 10


def build_optimizer(name, epochs, population):
    """Instantiate a registered Mealpy optimizer by short name."""
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    if population < MIN_POPULATION:
        raise ValueError(f"population must be >= {MIN_POPULATION} (Mealpy GA limit); got {population}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}")
    module_name, class_name, kwargs = ALGORITHMS[name]
    cls = getattr(getattr(mealpy, module_name), class_name)
    return cls(epoch=epochs, pop_size=population, **kwargs)


def decode_mask(position, threshold=0.5):
    """Continuous position -> boolean mask; never returns an empty mask.

    position: Shape [D], values in [0, 1].
    Returns: Shape [D], dtype bool.
    """
    position = np.asarray(position, dtype=float)
    mask = position >= threshold
    if not mask.any():
        # Keep the single most "selected" feature instead of an empty set.
        mask[int(np.argmax(position))] = True
    return mask


def default_classifier(k_neighbors=5):
    """sklearn kNN for the inner loop.

    The custom src/knn.KNN is validated to match sklearn's predictions,
    but it is far slower, and the inner loop runs thousands of fits.
    """
    return KNeighborsClassifier(n_neighbors=k_neighbors)


class FSObjective:
    """Callable fitness with a mask -> fitness cache (hash map).

    Many continuous positions decode to the same mask, especially late in a
    run. The cache skips refitting the classifier for masks already seen,
    while still counting every call so budgets stay comparable.
    """

    def __init__(self, X_train, y_train, X_valid, y_valid,
                 alpha=0.9, threshold=0.5, k_neighbors=5, classifier_factory=None):
        # X_train: Shape [N_train, D];  X_valid: Shape [N_valid, D]
        if X_train.shape[1] != X_valid.shape[1]:
            raise ValueError("Train and validation must have the same feature count.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.n_features = X_train.shape[1]
        self.alpha = alpha
        self.threshold = threshold
        self.make_clf = classifier_factory or (lambda: default_classifier(k_neighbors))
        self.cache = {}
        self.calls = 0
        self.failures = 0
        self.best_fitness = float("inf")
        self.best_mask = None
        self.best_f1 = None

    def score_mask(self, mask):
        """Return (fitness, macro_f1) for a boolean mask of Shape [D]."""
        model = self.make_clf().fit(self.X_train[:, mask], self.y_train)
        pred = model.predict(self.X_valid[:, mask])
        f1 = f1_score(self.y_valid, pred, average="macro", zero_division=0)
        ratio = mask.sum() / self.n_features
        return self.alpha * (1.0 - f1) + (1.0 - self.alpha) * ratio, float(f1)

    def __call__(self, position):
        self.calls += 1
        mask = decode_mask(position, self.threshold)
        key = mask.tobytes()
        if key not in self.cache:
            try:
                self.cache[key] = self.score_mask(mask)
            except Exception:
                self.failures += 1
                return 1.0  # worst possible fitness; not cached
        fitness, f1 = self.cache[key]
        if fitness < self.best_fitness:
            self.best_fitness, self.best_mask, self.best_f1 = float(fitness), mask.copy(), f1
        return float(fitness)


@dataclass
class FSResult:
    algorithm: str
    family: str
    mask: np.ndarray                 # Shape [D], dtype bool
    best_fitness: float
    best_inner_f1: float
    n_selected: int
    calls: int
    unique_masks: int
    failures: int
    runtime_seconds: float
    convergence: list = field(default_factory=list)  # best fitness per epoch

    @property
    def selected_indices(self):
        return np.flatnonzero(self.mask).astype(int)


def run_feature_selection(name, X_train, y_train, X_valid, y_valid,
                          epochs=10, population=10, seed=42,
                          alpha=0.9, threshold=0.5, k_neighbors=5,
                          classifier_factory=None, max_evaluations=None):
    """Run one metaheuristic on one inner split and return its best mask.

    Budget: by default the run stops after `epochs`. Algorithms spend a
    different number of fitness calls per epoch (e.g. HHO, ABC and SSA call
    the objective more often than PSO), so for a fair comparison pass
    `max_evaluations`. Mealpy checks it at the end of each epoch, so a run can
    overshoot by at most one epoch; the actual count is reported in `calls`.
    """
    objective = FSObjective(X_train, y_train, X_valid, y_valid, alpha=alpha,
                            threshold=threshold, k_neighbors=k_neighbors,
                            classifier_factory=classifier_factory)
    D = objective.n_features
    problem = {
        "obj_func": objective,
        "bounds": FloatVar(lb=[0.0] * D, ub=[1.0] * D),
        "minmax": "min",
        "log_to": None,
    }
    if max_evaluations is None:
        optimizer = build_optimizer(name, epochs, population)
        termination = None
    else:
        if max_evaluations < max(10, population):
            raise ValueError("max_evaluations must be >= max(10, population)")
        # Epoch cap is only a safety net; the evaluation budget stops the run.
        optimizer = build_optimizer(name, 10_000, population)
        termination = {"max_fe": int(max_evaluations)}
    start = time.perf_counter()
    optimizer.solve(problem, seed=seed, termination=termination)
    runtime = time.perf_counter() - start

    if objective.best_mask is None:
        raise RuntimeError(f"{name}: no successful fitness evaluation")
    return FSResult(
        algorithm=name,
        family=ALGORITHM_FAMILY.get(name, "other"),
        mask=objective.best_mask,
        best_fitness=objective.best_fitness,
        best_inner_f1=objective.best_f1,
        n_selected=int(objective.best_mask.sum()),
        calls=objective.calls,
        unique_masks=len(objective.cache),
        failures=objective.failures,
        runtime_seconds=round(runtime, 4),
        convergence=[float(v) for v in optimizer.history.list_global_best_fit],
    )
