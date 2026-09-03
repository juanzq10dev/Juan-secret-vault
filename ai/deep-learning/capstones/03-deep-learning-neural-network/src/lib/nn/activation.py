from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Activation(ABC):
    @abstractmethod
    def apply(self, Z: NDArray) -> NDArray:
        pass

    @abstractmethod
    def derivative(self, Z: NDArray) -> NDArray:
        pass


class ReLU(Activation):
    def apply(self, Z: NDArray) -> NDArray:
        return np.maximum(0, Z)

    def derivative(self, Z: NDArray) -> NDArray:
        return (Z > 0).astype(Z.dtype)


class Sigmoid(Activation):
    def apply(self, Z: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z: NDArray) -> NDArray:
        activation = self.apply(Z)
        return activation * (1 - activation)
