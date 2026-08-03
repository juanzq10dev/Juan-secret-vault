from abc import ABC, abstractmethod
from numpy.typing import NDArray


class TrainedModel(ABC):
    weights: NDArray
    bias: float

    @abstractmethod
    def predict(self, features: NDArray) -> float:
        pass


class UntrainedModel(ABC):
    @abstractmethod
    def fit(self, features: NDArray, outputs: NDArray) -> "TrainedModel":
        pass
