import numpy as np
from numpy.typing import NDArray


class NDArrayUtils:
    def get_features_count(self, ndarray: NDArray) -> int:
        return ndarray.shape[0]

    def get_samples_count(self, ndarray: NDArray) -> int:
        return ndarray.shape[1]

    def have_same_sample_count(self, first: NDArray, second: NDArray) -> bool:
        return self.get_samples_count(first) == self.get_samples_count(second)

    def randn(
        self,
        rows: int,
        cols: int,
        scale: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> NDArray:
        generator = rng if rng is not None else np.random.default_rng()
        return generator.standard_normal((rows, cols)) * scale

    def zeros(self, rows: int, cols: int) -> NDArray:
        return np.zeros((rows, cols))

    def as_column_major(self, ndarray: NDArray) -> NDArray:
        return ndarray.T


ndarray_utils = NDArrayUtils()
