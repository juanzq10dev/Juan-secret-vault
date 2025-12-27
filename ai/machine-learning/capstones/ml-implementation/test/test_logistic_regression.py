import numpy as np
import pytest
from regression_models.logistic_regression import (
    TrainedLogisticRegression,
    UntrainedLogisticRegression,
)


def test_fit_predict_binary_classification():
    # Arrange - Simple linearly separable binary data
    X = np.array([[-2], [-1], [1], [2]])
    y = np.array([0, 0, 1, 1])

    # Act
    model = UntrainedLogisticRegression().fit(X, y, learning_rate=0.1, epoch=1000)
    predictions = model.predict(X)

    # Assert - Should classify correctly
    predicted_classes = (predictions > 0.5).astype(int)
    np.testing.assert_array_equal(predicted_classes, y)


def test_fit_predict_probability_range():
    # Arrange
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])

    # Act
    model = UntrainedLogisticRegression().fit(X, y)
    predictions = model.predict(X)

    # Assert - All predictions should be between 0 and 1
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


def test_fit_predict_extreme_values():
    # Arrange - Test with more challenging data
    X = np.array([[-5], [-3], [3], [5]])
    y = np.array([0, 0, 1, 1])

    # Act
    model = UntrainedLogisticRegression().fit(X, y, learning_rate=0.01, epoch=2000)

    # Test on new data points
    test_predictions = model.predict(np.array([[-10], [0], [10]]))

    # Assert - Should show clear separation
    assert test_predictions[0] < 0.1  # Very negative should be close to 0
    assert 0.3 < test_predictions[1] < 0.7  # Around 0 should be around 0.5
    assert test_predictions[2] > 0.9  # Very positive should be close to 1


def test_fit_with_invalid_shapes_raises():
    model = UntrainedLogisticRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])  # Mismatched shape

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_trained_model_initialization():
    # Arrange
    weights = np.array([0.5, -0.3])
    bias = 0.1

    # Act
    model = TrainedLogisticRegression(weights, bias)

    # Assert
    np.testing.assert_array_equal(model.weights, weights)
    assert model.bias == bias


def test_predict_single_feature():
    # Arrange
    weights = np.array([2.0])
    bias = -1.0
    model = TrainedLogisticRegression(weights, bias)

    # Act
    prediction = model.predict([1.0])  # Should be sigmoid(2*1 - 1) = sigmoid(1)

    # Assert
    expected = 1 / (1 + np.exp(-1))  # sigmoid(1)
    assert pytest.approx(prediction, abs=1e-6) == expected
