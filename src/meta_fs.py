import time
from dataclasses import dataclass, field

import mealpy
import numpy as np
from mealpy import FloatVar
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

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

MIN_POPULATION = 10


def build_optimizer(name, epochs, population):
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    if population < MIN_POPULATION:
        raise ValueError(f"population must be >= {MIN_POPULATION} (Mealpy GA limit); got {population}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}")
    module_name, class_name, kwargs = ALGORITHMS[name]
    optimizer_class = getattr(getattr(mealpy, module_name), class_name)
    return optimizer_class(epoch=epochs, pop_size=population, **kwargs)


def decode_mask(position, threshold=0.5):
    position = np.asarray(position, dtype=float)
    mask = position >= threshold
    if not mask.any():
        mask[int(np.argmax(position))] = True
    return mask


def default_classifier(k_neighbors=5):
    return KNeighborsClassifier(n_neighbors=k_neighbors)


class FSObjective:
    def __init__(self, X_train, y_train, X_valid, y_valid,
                 alpha=0.9, threshold=0.5, k_neighbors=5, classifier_factory=None):
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
                return 1.0
        fitness, f1 = self.cache[key]
        if fitness < self.best_fitness:
            self.best_fitness, self.best_mask, self.best_f1 = float(fitness), mask.copy(), f1
        return float(fitness)


@dataclass
class FSResult:
    algorithm: str
    family: str
    mask: np.ndarray
    best_fitness: float
    best_inner_f1: float
    n_selected: int
    calls: int
    unique_masks: int
    failures: int
    runtime_seconds: float
    convergence: list = field(default_factory=list)

    @property
    def selected_indices(self):
        return np.flatnonzero(self.mask).astype(int)


def make_optimizer_and_stop(name, epochs, population, max_evaluations):
    if max_evaluations is None:
        return build_optimizer(name, epochs, population), None
    if max_evaluations < max(10, population):
        raise ValueError("max_evaluations must be >= max(10, population)")
    return build_optimizer(name, 10_000, population), {"max_fe": int(max_evaluations)}


def run_feature_selection(name, X_train, y_train, X_valid, y_valid,
                          epochs=10, population=10, seed=42,
                          alpha=0.9, threshold=0.5, k_neighbors=5,
                          classifier_factory=None, max_evaluations=None):
    objective = FSObjective(X_train, y_train, X_valid, y_valid, alpha=alpha,
                            threshold=threshold, k_neighbors=k_neighbors,
                            classifier_factory=classifier_factory)
    n = objective.n_features
    problem = {"obj_func": objective, "bounds": FloatVar(lb=[0.0] * n, ub=[1.0] * n),
               "minmax": "min", "log_to": None}
    optimizer, termination = make_optimizer_and_stop(name, epochs, population, max_evaluations)
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
