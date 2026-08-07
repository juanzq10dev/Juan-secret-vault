import argparse
from pathlib import Path

from numpy.typing import NDArray

from src.lib.managers.dataset_manager import dataset_manager
from src.lib.managers.metrics_manager import metrics
from src.lib.managers.plot_manager import plot_manager
from src.neural_network_models.shallow_neural_network import (
    TrainedShallowNeuralNetwork,
    UntrainedShallowNeuralNetwork,
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def train(
    X: NDArray,
    Y: NDArray,
    hidden_units: int,
    learning_rate: float = 1.2,
    epoch: int = 10_000,
    print_cost: bool = True,
) -> TrainedShallowNeuralNetwork:
    return UntrainedShallowNeuralNetwork(hidden_units=hidden_units, seed=3).fit(
        X, Y, learning_rate=learning_rate, epoch=epoch, print_cost=print_cost
    )


def evaluate(model: TrainedShallowNeuralNetwork, X: NDArray, Y: NDArray) -> float:
    accuracy = metrics.accuracy(model.predict(X), Y.astype(bool))
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


def sweep(X: NDArray, Y: NDArray) -> None:
    print(f"{'hidden units':>12} | {'accuracy':>8}")
    for hidden_units in (1, 2, 3, 4, 5):
        model = train(X, Y, hidden_units=hidden_units, print_cost=False)
        accuracy = metrics.accuracy(model.predict(X), Y.astype(bool))
        print(f"{hidden_units:>12} | {accuracy:>8.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planar data classification with one hidden layer"
    )
    parser.add_argument("--hidden-units", type=int, default=4)
    parser.add_argument(
        "--dataset",
        choices=["planar", "noisy_circles", "noisy_moons", "blobs", "gaussian_quantiles"],
        default="planar",
    )
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, Y = (
        dataset_manager.load_planar()
        if args.dataset == "planar"
        else dataset_manager.load_extra(args.dataset)
    )

    if args.sweep:
        sweep(X, Y)
        return

    model = train(X, Y, hidden_units=args.hidden_units)
    evaluate(model, X, Y)

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_manager.decision_boundary(
        model, X, Y, f"Decision boundary (hidden layer size {args.hidden_units})"
    )
    plot_manager.save(PLOTS_DIR / f"{args.dataset}_h{args.hidden_units}.png")


if __name__ == "__main__":
    main()
