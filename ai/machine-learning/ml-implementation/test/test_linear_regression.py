import numpy as np
import pytest
from regression_models.linear_regression import (
    TrainedLinearRegression,
    UntrainedLinearRegression,
)


def test_fit_predict_basic():
    # Arrange
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship

    # Act
    model = UntrainedLinearRegression().fit(X, y)
    predictions = model.predict(np.array([[6], [7]]))

    # Assert
    expected = np.array([12, 14])
    np.testing.assert_array_almost_equal(predictions, expected, decimal=5)


def test_fit_predict_with_bias():
    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])  # y = 2x + 1
    model = UntrainedLinearRegression().fit(X, y)
    predictions = model.predict(np.array([[4]]))
    assert pytest.approx(predictions[0], 0.001) == 9.0


def test_fit_with_invalid_shapes_raises():
    model = UntrainedLinearRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])  # Mismatched shape
    with pytest.raises(ValueError):
        model.fit(X, y)
