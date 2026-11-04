import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/"
N_FEATURES = 36


def landsat_path(name, data_dir=DATA_DIR):
    path = Path(data_dir) / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_URL + name, path)
    return path


def load_landsat_file(name, data_dir=DATA_DIR):
    data = np.loadtxt(landsat_path(name, data_dir))
    return data[:, :N_FEATURES], data[:, N_FEATURES].astype(int)


def load_landsat(data_dir=DATA_DIR):
    X_train, y_train = load_landsat_file("sat.trn", data_dir)
    X_test, y_test = load_landsat_file("sat.tst", data_dir)
    return X_train, y_train, X_test, y_test


def standardize(X_train, *others):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return [(X - mean) / std for X in (X_train, *others)]
