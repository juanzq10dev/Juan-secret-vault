import numpy as np
from numpy.typing import NDArray


class MetricsManager:
    def accuracy(self, predictions: NDArray, Y: NDArray) -> float:
        return float(np.mean(predictions == Y))


metrics = MetricsManager()
