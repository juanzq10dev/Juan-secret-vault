import sys

from numpy.typing import NDArray

from src.lib.managers.dataset_manager import dataset_manager
from src.regression_models.logistic_regression import (
    TrainedLogisticRegression,
    UntrainedLogisticRegression,
)


def accuracy(predictions: list[float], outputs: NDArray) -> float:
    predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
    correct = sum(p == y for p, y in zip(predicted_labels, outputs))
    return correct / len(outputs)


def train(
    train_x: NDArray, train_y: NDArray, test_x: NDArray, test_y: NDArray
) -> TrainedLogisticRegression:
    model = UntrainedLogisticRegression().fit(
        train_x, train_y, learning_rate=0.005, epoch=2000
    )

    train_predictions = [model.predict(x) for x in train_x]
    test_predictions = [model.predict(x) for x in test_x]

    print(f"Train accuracy: {accuracy(train_predictions, train_y):.2%}")
    print(f"Test accuracy: {accuracy(test_predictions, test_y):.2%}")
    return model


def classify_image(
    image_path: str, model: TrainedLogisticRegression, classes: NDArray
) -> None:
    image = dataset_manager.load_image(image_path)
    probability = model.predict(image)
    label = classes[1 if probability >= 0.5 else 0].decode("utf-8")
    print(f"{image_path} -> {label} ({probability:.2%} confidence)")


def main() -> None:
    train_x, train_y, test_x, test_y, classes = dataset_manager.load_catvnoncat()

    model = train(train_x, train_y, test_x, test_y)

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path:
        classify_image(image_path, model, classes)


if __name__ == "__main__":
    main()
