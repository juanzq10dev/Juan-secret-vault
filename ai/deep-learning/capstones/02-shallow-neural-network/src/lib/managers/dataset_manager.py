import numpy as np
import sklearn.datasets
from numpy.typing import NDArray

from src.lib.managers.numpy_manager import ndarray_utils


class DatasetManager:
    def load_planar(self, seed: int = 1) -> tuple[NDArray, NDArray]:
        return self._make_flower(seed)

    def load_extra(self, name: str, n_samples: int = 200) -> tuple[NDArray, NDArray]:
        if name == "noisy_circles":
            X, y = sklearn.datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.3)
        elif name == "noisy_moons":
            X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.2)
        elif name == "blobs":
            X, y = sklearn.datasets.make_blobs(
                n_samples=n_samples, random_state=5, n_features=2, centers=6
            )
            y = y % 2
        elif name == "gaussian_quantiles":
            X, y = sklearn.datasets.make_gaussian_quantiles(
                mean=None,
                cov=0.5,
                n_samples=n_samples,
                n_features=2,
                n_classes=2,
                shuffle=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

        return ndarray_utils.as_column_major(X), y.reshape(1, -1)

    def _make_flower(self, seed: int) -> tuple[NDArray, NDArray]:
        rng = np.random.default_rng(seed)
        m = 400
        samples_per_class = m // 2
        n_features = 2
        petals = 4
        radius_scale = 4

        X = np.zeros((m, n_features))
        Y = np.zeros(m, dtype=np.uint8)

        for class_label in range(2):
            ix = range(samples_per_class * class_label, samples_per_class * (class_label + 1))
            t = (
                np.linspace(class_label * 3.12, (class_label + 1) * 3.12, samples_per_class)
                + rng.standard_normal(samples_per_class) * 0.2
            )
            r = radius_scale * np.sin(petals * t) + rng.standard_normal(samples_per_class) * 0.2
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = class_label

        return ndarray_utils.as_column_major(X), Y.reshape(1, -1)


dataset_manager = DatasetManager()
