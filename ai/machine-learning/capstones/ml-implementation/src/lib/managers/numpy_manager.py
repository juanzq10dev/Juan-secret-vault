import numpy as np
from numpy.typing import NDArray


class NDArrayUtils:
    def from_list(self, list: list) -> NDArray:
        return np.array(list)

    def zeros(self, size: int) -> NDArray:
        return np.zeros(size)

    def get_rows_count(self, ndarray: NDArray) -> int:
        return ndarray.shape[0]

    def get_columns_count(self, ndarray: NDArray) -> int:
        return ndarray.shape[1]

    def have_same_row_count(self, first: NDArray, second: NDArray) -> bool:
        return self.get_rows_count(first) == self.get_rows_count(second)


ndarray_utils = NDArrayUtils()
