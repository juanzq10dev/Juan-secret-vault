from src.lib.strategies.logistic_regression_strategy import LogisticRegressionStrategy
from src.lib.contexts.supervised_learning_model_context import (
    TrainedRegression,
    UntrainedRegression,
)
from numpy.typing import NDArray


class TrainedLogisticRegression(TrainedRegression):
    def __init__(self, weights: NDArray, bias: float) -> None:
        super().__init__(weights, bias, LogisticRegressionStrategy())


class UntrainedLogisticRegression(UntrainedRegression):
    def __init__(self) -> None:
        super().__init__(LogisticRegressionStrategy())

    def fit(
        self,
        features: NDArray,
        outputs: NDArray,
        learning_rate: float = 0.1,
        epoch: int = 1000,
    ) -> TrainedLogisticRegression:
        trained_model = super().fit(features, outputs, learning_rate, epoch)
        return TrainedLogisticRegression(trained_model.weights, trained_model.bias)
