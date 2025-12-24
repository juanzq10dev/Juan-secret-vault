import numpy as np
from lib.abc.model_strategy import ModelStrategy
from numpy.typing import NDArray


class LinearRegressionStrategy(ModelStrategy):
    def model(self, x: NDArray, w: NDArray, b: float) -> float:
        return np.dot(x, w) + b

    def cost_function(self, y_hat: NDArray, y: NDArray) -> float:
        if y_hat.shape != y.shape:
            raise ValueError("Shapes of y_hat and y must be equal")

        m: int = y_hat.shape[0]

        return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

    def gradient_descent(
        self,
        x: NDArray,
        y: NDArray,
        w: NDArray,
        b: float,
        learning_rate: float,
        epoch: int,
    ) -> tuple[NDArray, float]:
        m = x.shape[0]

        for i in range(epoch):
            y_hat = self.model(x, w, b)

            dw = (1 / m) * np.dot(x.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y)

            w = w - learning_rate * dw
            b = b - learning_rate * db

        return w, b
