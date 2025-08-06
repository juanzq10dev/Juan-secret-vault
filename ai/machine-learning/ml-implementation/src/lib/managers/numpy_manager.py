import numpy as np


class NumpyManager:
    def create_ndarray_from_list(self, list: list) -> np.ndarray:
        return np.array(list)

    def create_zeros_array_with_len(self, len: int) -> np.ndarray:
        return np.zeros(len)

    def ndarray_rows(self, ndarray: np.ndarray) -> int:
        return ndarray.shape[0]

    def ndarray_columns(self, ndarray: np.ndarray) -> int:
        return ndarray.shape[1]

    def have_same_rows(self, first: np.ndarray, second: np.ndarray) -> bool:
        return self.ndarray_rows(first) == self.ndarray_rows(second)
