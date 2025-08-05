import numpy as np
from lib.math.functions import gradient_descent, linear_regression_model
from lib.managers.numpy_manager import NumpyManager


class LinearRegression:
    def __init__(self):
        self.weights = None 
        self.bias = 0
        self.fitted = False
        self.numpy_manager = NumpyManager()

    def fit(self, features: list, outputs: list):
        X: np.ndarray = self.numpy_manager.create_ndarray_from_list(features)
        y: np.ndarray  = self.numpy_manager.create_ndarray_from_list(outputs)

        if not self.numpy_manager.have_same_rows(X, y):
            raise ValueError("Number of samples in X and y must match.")

        m = self.numpy_manager.ndarray_columns(features)
        self.weights = self.numpy_manager.create_zeros_array_with_len(m)
        
        w, b = gradient_descent(X, y, self.weights, self.bias, 0.1, 1000)
        self.weights = w
        self.bias = b
        self.fitted = True

        return self.fitted

    def predict(self, features: list):
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        X = np.array(features)
        return linear_regression_model(X, self.weights, self.bias)
