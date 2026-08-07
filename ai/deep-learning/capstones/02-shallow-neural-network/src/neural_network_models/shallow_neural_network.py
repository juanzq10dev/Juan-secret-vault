from numpy.typing import NDArray

from src.lib.contexts.neural_network_context import (
    TrainedNeuralNetwork,
    UntrainedNeuralNetwork,
)
from src.lib.strategies.shallow_neural_network_strategy import (
    ShallowNeuralNetworkStrategy,
)
from src.lib.types.network import Parameters


class TrainedShallowNeuralNetwork(TrainedNeuralNetwork):
    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters, ShallowNeuralNetworkStrategy())


class UntrainedShallowNeuralNetwork(UntrainedNeuralNetwork):
    def __init__(self, hidden_units: int = 4, seed: int | None = 3) -> None:
        super().__init__(ShallowNeuralNetworkStrategy(seed), hidden_units)

    def fit(
        self,
        features: NDArray,
        outputs: NDArray,
        learning_rate: float = 1.2,
        epoch: int = 10_000,
        print_cost: bool = False,
    ) -> TrainedShallowNeuralNetwork:
        trained_model = super().fit(features, outputs, learning_rate, epoch, print_cost)
        return TrainedShallowNeuralNetwork(trained_model.parameters)
