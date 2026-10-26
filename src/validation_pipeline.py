
"""Reproducible, leakage-safe validation splits for numeric datasets."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__:
    from .dataset_loader import LoadedDataset
else:
    # Scripts that add ``src`` to sys.path import this module directly.
    from dataset_loader import LoadedDataset


@dataclass
class ValidationData:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    scaler: StandardScaler
    train_indices: np.ndarray
    valid_indices: np.ndarray
    seed: int
    validation_fraction: float


def make_validation_split(
    dataset: LoadedDataset,
    validation_fraction: float = 0.2,
    seed: int = 42,
    scale: bool = True,
) -> ValidationData:
    """Split a classification dataset and fit preprocessing on train only."""

    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target array.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(y) < 2:
        raise ValueError("At least two rows are required.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Classification requires at least two classes.")
    if counts.min() < 2:
        raise ValueError(
            "Each class needs at least two examples for a stratified split."
        )

    indices = np.arange(len(y))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y,
    )

    X_train = X[train_idx].copy()
    X_valid = X[valid_idx].copy()

    # Fit only on training rows. Validation rows never affect the scaler.
    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
    else:
        # Still fit for consistent metadata; do not transform either split.
        scaler.fit(X_train)

    return ValidationData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y[train_idx].copy(),
        y_valid=y[valid_idx].copy(),
        scaler=scaler,
        train_indices=train_idx,
        valid_indices=valid_idx,
        seed=seed,
        validation_fraction=validation_fraction,
    )
