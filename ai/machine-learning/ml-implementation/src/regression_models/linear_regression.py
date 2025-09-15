from lib.math.functions import gradient_descent, linear_regression_model
from lib.managers.numpy_manager import ndarray_utils
from lib.abc.models import TrainedModel, UntrainedModel
from numpy.typing import NDArray


class TrainedLinearRegression(TrainedModel):
    def __init__(self, weights: NDArray, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def predict(self, features: list) -> float:
        X = ndarray_utils.from_list(features)
        return linear_regression_model(X, self.weights, self.bias)


class UntrainedLinearRegression(UntrainedModel):
    def __init__(self) -> None:
        pass

    def fit(
        self,
        features: list,
        outputs: list,
        learning_rate: float = 0.1,
        epoch: int = 1000,
    ) -> "TrainedModel":
        X: NDArray = ndarray_utils.from_list(features)
        y: NDArray = ndarray_utils.from_list(outputs)

        if not ndarray_utils.have_same_row_count(X, y):
            raise ValueError("Number of samples in X and y must match.")

        m = ndarray_utils.get_columns_count(X)
        initial_weights = ndarray_utils.zeros(m)
        initial_bias = 0

        weights, bias = gradient_descent(
            X, y, initial_weights, initial_bias, learning_rate, epoch
        )
        return TrainedLinearRegression(weights, bias)
