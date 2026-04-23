from lib.abc.model_strategy import ModelStrategy
from lib.managers.numpy_manager import ndarray_utils
from lib.abc.models import TrainedModel, UntrainedModel
from numpy.typing import NDArray


class TrainedRegression(TrainedModel):
    def __init__(self, weights: NDArray, bias: float, strategy: ModelStrategy) -> None:
        self.weights = weights
        self.bias = bias
        self.strategy = strategy

    def predict(self, features: list) -> float:
        X = ndarray_utils.from_list(features)
        return self.strategy.model(X, self.weights, self.bias)


class UntrainedRegression(UntrainedModel):
    def __init__(self, strategy: ModelStrategy) -> None:
        self.strategy = strategy

    def fit(
        self,
        features: list,
        outputs: list,
        learning_rate: float = 0.1,
        epoch: int = 1000,
    ) -> TrainedModel:
        X: NDArray = ndarray_utils.from_list(features)
        y: NDArray = ndarray_utils.from_list(outputs)

        if not ndarray_utils.have_same_row_count(X, y):
            raise ValueError("Number of samples in X and y must match.")

        m = ndarray_utils.get_columns_count(X)
        initial_weights = ndarray_utils.zeros(m)
        initial_bias = 0

        weights, bias = self.strategy.gradient_descent(
            X, y, initial_weights, initial_bias, learning_rate, epoch
        )
        return TrainedRegression(weights, bias, self.strategy)
