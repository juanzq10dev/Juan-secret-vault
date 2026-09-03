import argparse
from pathlib import Path

from numpy.typing import NDArray

from src.lib.managers.dataset_manager import dataset_manager
from src.lib.managers.metrics_manager import metrics
from src.lib.managers.plot_manager import plot_manager
from src.neural_network_models.deep_neural_network import (
    TrainedDeepNeuralNetwork,
    UntrainedDeepNeuralNetwork,
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def evaluate(model: TrainedDeepNeuralNetwork, X: NDArray, Y: NDArray, split: str) -> float:
    accuracy = metrics.accuracy(model.predict(X), Y.astype(bool))
    print(f"{split} accuracy: {accuracy:.2%}")
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L-layer deep neural network for cat vs non-cat classification"
    )
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[20, 7, 5])
    parser.add_argument("--learning-rate", type=float, default=0.0075)
    parser.add_argument("--epochs", type=int, default=2_500)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_x, train_y, test_x, test_y, _ = dataset_manager.load_cats()

    model = UntrainedDeepNeuralNetwork(hidden_layers=args.hidden_layers, seed=1).fit(
        train_x,
        train_y,
        learning_rate=args.learning_rate,
        epoch=args.epochs,
        print_cost=not args.quiet,
    )

    evaluate(model, train_x, train_y, "Train")
    evaluate(model, test_x, test_y, "Test")

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_manager.cost_curve(model.costs, args.learning_rate)
    layers = "-".join(str(units) for units in args.hidden_layers)
    plot_manager.save(PLOTS_DIR / f"cost_{layers}.png")


if __name__ == "__main__":
    main()
