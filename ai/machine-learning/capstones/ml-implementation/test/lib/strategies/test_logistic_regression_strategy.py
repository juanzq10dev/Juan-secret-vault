import numpy as np
import pytest
from lib.strategies.logistic_regression_strategy import LogisticRegressionStrategy


def test_model():
    strategy = LogisticRegressionStrategy()
    x = np.array([[0], [1], [2]])
    w = np.array([1.0])
    b = 0.0

    result = strategy.model(x, w, b)
    expected = np.array([0.5, 0.7311, 0.8808])  # sigmoid(0), sigmoid(1), sigmoid(2)

    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_cost_function():
    strategy = LogisticRegressionStrategy()
    y_hat = np.array([0.7, 0.3, 0.8])
    y = np.array([1, 0, 1])

    cost = strategy.cost_function(y_hat, y)
    expected = np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    assert pytest.approx(cost, abs=1e-6) == expected


def test_cost_function_with_mismatched_shapes_raises():
    strategy = LogisticRegressionStrategy()
    y_hat = np.array([0.7, 0.3, 0.8])
    y = np.array([1, 0])

    with pytest.raises(ValueError):
        strategy.cost_function(y_hat, y)


def test_gradient_descent_binary_classification():
    strategy = LogisticRegressionStrategy()
    # Simple linearly separable data
    x = np.array([[-2], [-1], [1], [2]])
    y = np.array([0, 0, 1, 1])
    w = np.array([0.0])
    b = 0.0
    learning_rate = 0.1
    epoch = 1000

    new_w, new_b = strategy.gradient_descent(x, y, w, b, learning_rate, epoch)

    # Test predictions on training data
    predictions = strategy.model(x, new_w, new_b)
    predicted_classes = (predictions > 0.5).astype(int)

    # Should achieve reasonable classification
    np.testing.assert_array_equal(predicted_classes, y)
