from abc import ABC, abstractmethod

from numpy.typing import NDArray

from src.lib.types.network import Parameters


class TrainedModel(ABC):
    parameters: Parameters

    @abstractmethod
    def predict(self, features: NDArray) -> NDArray:
        pass


class UntrainedModel(ABC):
    @abstractmethod
    def fit(self, features: NDArray, outputs: NDArray) -> "TrainedModel":
        pass
