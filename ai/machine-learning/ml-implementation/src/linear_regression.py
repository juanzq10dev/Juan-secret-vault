import numpy as np
from lib.math.functions import gradient_descent, linear_regression_model, squared_error_cost_function


class LinearRegression:
    def __init__(self):
        self.weights = None 
        self.bias = 0
        self.fitted = False

    def fit(self, X: list, y: list):
        X = np.array(X)
        y = np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        m = X.shape[1]
        self.weights = np.zeros(m)
        
        w, b = gradient_descent(X, y, self.weights, self.bias, 0.1, 1000)
        self.weights = w
        self.bias = b
        self.fitted = True

        return self.fitted

    def predict(self, X: list):
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        X = np.array(X)
        return linear_regression_model(X, self.weights, self.bias)
