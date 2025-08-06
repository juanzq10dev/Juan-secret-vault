import numpy as np
from lib.math.functions import gradient_descent, linear_regression_model
from lib.managers.numpy_manager import NumpyManager
from lib.abc.models import TrainedModel, UntrainedModel


class TrainedLinearRegression(TrainedModel):
    def __init__(self, weights: np.ndarray, bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.numpy_manager = NumpyManager()

    def predict(self, features: list) -> float:
        X = self.numpy_manager.create_ndarray_from_list(features)
        return linear_regression_model(X, self.weights, self.bias)


class UntrainedLinearRegression(UntrainedModel):
    def __init__(self) -> None:
        self.numpy_manager = NumpyManager()

    def fit(self, features: list, outputs: list) -> "TrainedModel":
        X: np.ndarray = self.numpy_manager.create_ndarray_from_list(features)
        y: np.ndarray = self.numpy_manager.create_ndarray_from_list(outputs)

        if not self.numpy_manager.have_same_rows(X, y):
            raise ValueError("Number of samples in X and y must match.")

        m = self.numpy_manager.ndarray_columns(X)
        initial_weights = self.numpy_manager.create_zeros_array_with_len(m)
        initial_bias = 0

        weights, bias = gradient_descent(X, y, initial_weights, initial_bias, 0.1, 1000)
        return TrainedLinearRegression(weights, bias)
