
"""Configurable loader for numeric machine-learning datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    path: str
    delimiter: str = r"\s+"
    header: Any = None
    target_column: Any = -1
    feature_columns: list | None = None
    task: str = "classification"


@dataclass
class LoadedDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    class_labels: list | None
    source_path: str


def load_dataset(config: DatasetConfig) -> LoadedDataset:
    """Load a dataset and validate its features and target."""

    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if config.task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")

    df = pd.read_csv(
        path,
        sep=config.delimiter,
        header=config.header,
        engine="python",
    )

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain features and a target column.")

    # Resolve the target by column name or integer position.
    target = config.target_column
    if isinstance(target, int):
        target_index = target if target >= 0 else len(df.columns) + target
        if not 0 <= target_index < len(df.columns):
            raise ValueError("target_column index is out of range.")
        target_name = df.columns[target_index]
    else:
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")
        target_name = target

    if config.feature_columns is None:
        feature_names = [
            col for col in df.columns if col != target_name
        ]
    else:
        feature_names = config.feature_columns
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        if target_name in feature_names:
            raise ValueError("Target cannot also be a feature.")

    if not feature_names:
        raise ValueError("No feature columns were selected.")

    X_df = df[feature_names].apply(pd.to_numeric, errors="raise")
    y_series = df[target_name]

    if X_df.isna().any().any():
        raise ValueError(
            "Features contain missing values. Configure imputation "
            "in the preprocessing pipeline before using this dataset."
        )
    if y_series.isna().any():
        raise ValueError("Target column contains missing values.")

    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Features contain infinite or non-finite values.")

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    if config.task == "classification":
        labels = pd.unique(y).tolist()
        if len(labels) < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )
        class_labels = sorted(labels, key=str)
    else:
        y = pd.to_numeric(pd.Series(y), errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Regression target contains non-finite values.")
        class_labels = None

    return LoadedDataset(
        X=X,
        y=y,
        feature_names=[str(c) for c in feature_names],
        target_name=str(target_name),
        class_labels=class_labels,
        source_path=str(path.resolve()),
    )
