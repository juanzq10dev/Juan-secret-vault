from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Loss(ABC):
    @abstractmethod
    def cost(self, AL: NDArray, Y: NDArray) -> float:
        pass

    @abstractmethod
    def gradient(self, AL: NDArray, Y: NDArray) -> NDArray:
        pass


class BinaryCrossEntropy(Loss):
    def cost(self, AL: NDArray, Y: NDArray) -> float:
        m = Y.shape[1]
        losses = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
        return float(-np.sum(losses) / m)

    def gradient(self, AL: NDArray, Y: NDArray) -> NDArray:
        return -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
