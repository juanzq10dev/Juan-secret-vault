import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class ModelStrategy(ABC):
    @abstractmethod
    def model(self, x: NDArray, w: NDArray, b: float) -> float:
        pass

    @abstractmethod
    def cost_function(self, y_hat: NDArray, y: NDArray) -> float:
        pass

    @abstractmethod
    def gradient_descent(
        self,
        x: NDArray,
        y: NDArray,
        w: NDArray,
        b: float,
        learning_rate: float,
        epoch: int,
    ) -> tuple[NDArray, float]:
        pass
