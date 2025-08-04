import numpy as np
import pytest
from lib.math.functions import linear_regression_model, squared_error_cost_function


def test_basic_case() -> None:
    x = np.array([1, 2])
    w = np.array([3, 4])
    b = 5
    result = linear_regression_model(x, w, b)
    expected = 1 * 3 + 2 * 4 + 5  # = 16
    assert result == expected


def test_zero_bias() -> None:
    x = np.array([1, 2, 3])
    w = np.array([0.5, 0.5, 0.5])
    b = 0
    result = linear_regression_model(x, w, b)
    expected = 3.0
    assert result == pytest.approx(expected)


def test_array_input() -> None:
    x = np.array([[1, 2], [3, 4]])
    w = np.array([0.5, 1.5])
    b = 1.0
    result = linear_regression_model(x, w, b)
    expected = np.dot(x, w) + b
    np.testing.assert_array_almost_equal(result, expected)


def test_invalid_shape() -> None:
    x = np.array([1, 2])
    w = np.array([1, 2, 3])  # Mismatched shape
    b = 0
    with pytest.raises(ValueError):
        linear_regression_model(x, w, b)


## Test squared error cost function


def test_squared_error_cost_function_correctness():
    y_hat = np.array([2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0])

    expected = 0.5
    result = squared_error_cost_function(y_hat, y)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_squared_error_cost_function_shape_mismatch():
    y_hat = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError) as exc_info:
        squared_error_cost_function(y_hat, y)
    assert "Shapes of y_hat and y must be equal" in str(exc_info.value)


def test_squared_error_cost_function_zero_error():
    y_hat = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    result = squared_error_cost_function(y_hat, y)
    assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"
