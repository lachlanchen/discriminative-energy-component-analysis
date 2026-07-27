"""Reproducible public and synthetic datasets for DECA experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO, TextIOWrapper
from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.io import arff
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """A numerical classification dataset and its provenance."""

    name: str
    X: FloatArray
    y: NDArray
    source: str
    kind: str


UCI_ARCHIVES = {
    "banknote": {
        "url": (
            "https://archive.ics.uci.edu/static/public/267/"
            "banknote+authentication.zip"
        ),
        "sha256": (
            "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227"
        ),
    },
    "letter": {
        "url": (
            "https://archive.ics.uci.edu/static/public/59/"
            "letter+recognition.zip"
        ),
        "sha256": (
            "3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f"
        ),
    },
    "dry_bean": {
        "url": (
            "https://archive.ics.uci.edu/static/public/602/"
            "dry+bean+dataset.zip"
        ),
        "sha256": (
            "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
        ),
    },
    "spambase": {
        "url": (
            "https://archive.ics.uci.edu/static/public/94/spambase.zip"
        ),
        "sha256": (
            "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636"
        ),
    },
}


def _download_verified(name: str, data_home: Path) -> bytes:
    metadata = UCI_ARCHIVES[name]
    data_home.mkdir(parents=True, exist_ok=True)
    destination = data_home / f"{name}.zip"
    if destination.exists():
        payload = destination.read_bytes()
    else:
        with urlopen(metadata["url"], timeout=120) as response:
            payload = response.read()
        destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {name}: expected "
            f"{metadata['sha256']}, received {digest}."
        )
    return payload


def _load_uci(name: str, data_home: Path) -> Dataset:
    payload = _download_verified(name, data_home)
    source = UCI_ARCHIVES[name]["url"]
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if name == "banknote":
            frame = pd.read_csv(
                archive.open("data_banknote_authentication.txt"),
                header=None,
            )
            X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
            y = frame.iloc[:, -1].to_numpy()
        elif name == "letter":
            frame = pd.read_csv(
                archive.open("letter-recognition.data"), header=None
            )
            X = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
            y = frame.iloc[:, 0].to_numpy()
        elif name == "dry_bean":
            with archive.open(
                "DryBeanDataset/Dry_Bean_Dataset.arff"
            ) as binary_stream:
                raw, _ = arff.loadarff(
                    TextIOWrapper(binary_stream, encoding="utf-8")
                )
            frame = pd.DataFrame(raw)
            X = frame.drop(columns=["Class"]).to_numpy(dtype=np.float64)
            y = frame["Class"].str.decode("utf-8").to_numpy()
        elif name == "spambase":
            frame = pd.read_csv(archive.open("spambase.data"), header=None)
            X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
            y = frame.iloc[:, -1].to_numpy()
        else:
            raise ValueError(f"Unknown UCI dataset {name}.")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{name} contains non-finite numerical features.")
    return Dataset(name, X, y, source, "public_tabular")


def _sklearn_datasets() -> dict[str, Dataset]:
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    datasets = {}
    for name, loader in loaders.items():
        bunch = loader()
        datasets[name] = Dataset(
            name=name,
            X=np.asarray(bunch.data, dtype=np.float64),
            y=np.asarray(bunch.target),
            source=f"scikit-learn built-in: {name}",
            kind="public_tabular" if name != "digits" else "public_image",
        )
    return datasets


def covariance_signal(
    samples_per_class: int = 1000,
    dimension: int = 12,
    random_state: int = 1729,
) -> Dataset:
    """Zero-mean classes that differ only in covariance."""

    if dimension < 4:
        raise ValueError("dimension must be at least four.")
    rng = np.random.default_rng(random_state)
    base = np.ones(dimension)
    eigenvalues_0 = base.copy()
    eigenvalues_1 = base.copy()
    eigenvalues_0[:2] = (4.0, 0.25)
    eigenvalues_1[:2] = (0.25, 4.0)
    X0 = rng.normal(size=(samples_per_class, dimension)) * np.sqrt(
        eigenvalues_0
    )
    X1 = rng.normal(size=(samples_per_class, dimension)) * np.sqrt(
        eigenvalues_1
    )
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], samples_per_class)
    return Dataset(
        "synthetic_covariance",
        X,
        y,
        "generated by deca.datasets.covariance_signal",
        "synthetic_covariance",
    )


def mean_signal(
    samples_per_class: int = 1000,
    dimension: int = 12,
    random_state: int = 2718,
) -> Dataset:
    """Equal-covariance classes separated by their means."""

    rng = np.random.default_rng(random_state)
    mean = np.zeros(dimension)
    mean[0] = 1.0
    X0 = rng.normal(scale=1.0, size=(samples_per_class, dimension)) - mean
    X1 = rng.normal(scale=1.0, size=(samples_per_class, dimension)) + mean
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], samples_per_class)
    return Dataset(
        "synthetic_mean",
        X,
        y,
        "generated by deca.datasets.mean_signal",
        "synthetic_mean",
    )


def benchmark_datasets(
    data_home: str | Path,
    include_large: bool = True,
) -> dict[str, Dataset]:
    """Return fixed benchmark datasets with verified public archives."""

    data_path = Path(data_home)
    datasets = {
        "synthetic_covariance": covariance_signal(),
        "synthetic_mean": mean_signal(),
        **_sklearn_datasets(),
        "banknote": _load_uci("banknote", data_path),
        "spambase": _load_uci("spambase", data_path),
    }
    if include_large:
        datasets["dry_bean"] = _load_uci("dry_bean", data_path)
        datasets["letter"] = _load_uci("letter", data_path)
    return datasets
