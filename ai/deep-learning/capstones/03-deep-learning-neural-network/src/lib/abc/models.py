from abc import ABC, abstractmethod

from numpy.typing import NDArray


class TrainedModel(ABC):
    @abstractmethod
    def predict(self, features: NDArray) -> NDArray:
        pass


class UntrainedModel(ABC):
    @abstractmethod
    def fit(self, features: NDArray, outputs: NDArray) -> "TrainedModel":
        pass
