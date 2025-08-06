from abc import ABC, abstractmethod


class TrainedModel(ABC):
    @abstractmethod
    def predict(self, features: list) -> float:
        pass


class UntrainedModel(ABC):
    @abstractmethod
    def fit(self, features: list, outputs: list) -> "TrainedModel":
        pass
