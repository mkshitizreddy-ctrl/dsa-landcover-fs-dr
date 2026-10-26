
import numpy as np
import pandas as pd
import pytest

from src.dataset_loader import DatasetConfig, load_dataset
from src.validation_pipeline import make_validation_split


def test_load_landsat_dataset():
    dataset = load_dataset(DatasetConfig(path="data/sat.trn"))

    assert dataset.X.shape == (4435, 36)
    assert len(dataset.y) == 4435
    assert dataset.class_labels == [1, 2, 3, 4, 5, 7]
    assert np.isfinite(dataset.X).all()


def test_load_csv_with_named_target(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6],
        "feature_b": [10, 20, 30, 40, 50, 60],
        "label": ["A", "A", "B", "B", "A", "B"],
    }).to_csv(path, index=False)

    dataset = load_dataset(DatasetConfig(
        path=str(path),
        delimiter=",",
        header=0,
        target_column="label",
    ))

    assert dataset.X.shape == (6, 2)
    assert dataset.feature_names == ["feature_a", "feature_b"]
    assert dataset.target_name == "label"
    assert set(dataset.class_labels) == {"A", "B"}


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_dataset(DatasetConfig(path="missing_dataset.csv"))


def test_validation_split_is_reproducible():
    dataset = load_dataset(DatasetConfig(path="data/sat.trn"))

    first = make_validation_split(dataset, seed=42)
    second = make_validation_split(dataset, seed=42)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.valid_indices, second.valid_indices)


def test_split_has_no_overlap_and_preserves_classes():
    dataset = load_dataset(DatasetConfig(path="data/sat.trn"))
    split = make_validation_split(dataset, validation_fraction=0.2, seed=42)

    assert len(split.train_indices) == 3548
    assert len(split.valid_indices) == 887
    assert set(split.train_indices).isdisjoint(set(split.valid_indices))
    assert set(split.y_train) == set(dataset.class_labels)
    assert set(split.y_valid) == set(dataset.class_labels)


def test_scaler_is_fitted_on_training_rows_only():
    dataset = load_dataset(DatasetConfig(path="data/sat.trn"))
    split = make_validation_split(dataset, seed=42)

    expected_mean = dataset.X[split.train_indices].mean(axis=0)
    expected_scale = dataset.X[split.train_indices].std(axis=0)

    np.testing.assert_allclose(split.scaler.mean_, expected_mean)
    np.testing.assert_allclose(split.scaler.scale_, expected_scale)

    # Standardized training means should be approximately zero.
    np.testing.assert_allclose(
        split.X_train.mean(axis=0), 0.0, atol=1e-10
    )


def test_invalid_validation_fraction_raises_error():
    dataset = load_dataset(DatasetConfig(path="data/sat.trn"))

    with pytest.raises(ValueError):
        make_validation_split(dataset, validation_fraction=1.0)
