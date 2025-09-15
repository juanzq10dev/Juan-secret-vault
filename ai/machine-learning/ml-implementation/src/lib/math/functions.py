import numpy as np
from numpy.typing import NDArray


def linear_regression_model(x: NDArray, w: NDArray, b: float) -> float:
    return np.dot(x, w) + b


def squared_error_cost_function(y_hat: NDArray, y: NDArray) -> float:
    if y_hat.shape != y.shape:
        raise ValueError("Shapes of y_hat and y must be equal")

    m: int = y_hat.shape[0]

    return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)


def gradient_descent(
    x: NDArray,
    y: NDArray,
    w: NDArray,
    b: float,
    learning_rate: float,
    epoch: int,
) -> tuple[NDArray, float]:
    m = x.shape[0]

    for i in range(epoch):
        y_hat = linear_regression_model(x, w, b)

        dw = (1 / m) * np.dot(x.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b
