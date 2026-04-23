from lib.strategies.linear_regression_strategy import LinearRegressionStrategy
from lib.contexts.supervised_learning_model_context import (
    TrainedRegression,
    UntrainedRegression,
)
from lib.abc.models import TrainedModel


class TrainedLinearRegression(TrainedRegression):
    def __init__(self, weights, bias) -> None:
        super().__init__(weights, bias, LinearRegressionStrategy())


class UntrainedLinearRegression(UntrainedRegression):
    def __init__(self) -> None:
        super().__init__(LinearRegressionStrategy())

    def fit(
        self,
        features: list,
        outputs: list,
        learning_rate: float = 0.1,
        epoch: int = 1000,
    ) -> TrainedModel:
        trained_model = super().fit(features, outputs, learning_rate, epoch)
        return TrainedLinearRegression(trained_model.weights, trained_model.bias)
