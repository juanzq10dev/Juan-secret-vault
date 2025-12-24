from lib.strategies.logistic_regression_strategy import LogisticRegressionStrategy
from lib.contexts.supervised_learning_model_context import (
    TrainedRegression,
    UntrainedRegression,
)
from lib.abc.models import TrainedModel


class TrainedLogisticRegression(TrainedRegression):
    def __init__(self, weights, bias) -> None:
        super().__init__(weights, bias, LogisticRegressionStrategy())


class UntrainedLogisticRegression(UntrainedRegression):
    def __init__(self) -> None:
        super().__init__(LogisticRegressionStrategy())

    def fit(
        self,
        features: list,
        outputs: list,
        learning_rate: float = 0.1,
        epoch: int = 1000,
    ) -> TrainedModel:
        trained_model = super().fit(features, outputs, learning_rate, epoch)
        return TrainedLogisticRegression(trained_model.weights, trained_model.bias)
