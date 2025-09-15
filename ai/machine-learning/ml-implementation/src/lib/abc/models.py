from abc import ABC, abstractmethod
from numpy.typing import NDArray


class TrainedModel(ABC):
    weights: NDArray
    bias: float

    @abstractmethod
    def predict(self, features: list) -> float:
        pass


class UntrainedModel(ABC):
    @abstractmethod
    def fit(self, features: list, outputs: list) -> "TrainedModel":
        pass
