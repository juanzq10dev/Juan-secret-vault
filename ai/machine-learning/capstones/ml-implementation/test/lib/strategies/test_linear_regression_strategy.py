import numpy as np
import pytest
from lib.strategies.linear_regression_strategy import LinearRegressionStrategy


def test_model():
    strategy = LinearRegressionStrategy()
    x = np.array([[1, 2], [3, 4]])
    w = np.array([0.5, 1.0])
    b = 0.1

    result = strategy.model(x, w, b)
    expected = np.array([2.6, 5.6])  # [1*0.5 + 2*1.0 + 0.1, 3*0.5 + 4*1.0 + 0.1]

    np.testing.assert_array_almost_equal(result, expected)


def test_cost_function():
    strategy = LinearRegressionStrategy()
    y_hat = np.array([2.0, 4.0, 6.0])
    y = np.array([2.1, 3.9, 6.1])

    cost = strategy.cost_function(y_hat, y)
    expected = 0.5 * np.mean((y_hat - y) ** 2)

    assert pytest.approx(cost, abs=1e-6) == expected


def test_cost_function_with_mismatched_shapes_raises():
    strategy = LinearRegressionStrategy()
    y_hat = np.array([1, 2, 3])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        strategy.cost_function(y_hat, y)


def test_gradient_descent():
    strategy = LinearRegressionStrategy()
    x = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    w = np.array([0.0])
    b = 0.0
    learning_rate = 0.01
    epoch = 1000

    new_w, new_b = strategy.gradient_descent(x, y, w, b, learning_rate, epoch)

    # Should converge to approximately w=2, b=0 for y=2x
    assert pytest.approx(new_w[0], abs=0.2) == 2.0
    assert pytest.approx(new_b, abs=0.2) == 0.0
