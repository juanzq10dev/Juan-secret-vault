from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.lib.contexts.neural_network_context import (
    TrainedNeuralNetwork,
    UntrainedNeuralNetwork,
)
from src.lib.nn import (
    BinaryCrossEntropy,
    DenseLayer,
    Layer,
    NeuralNetwork,
    ReLU,
    Sigmoid,
)


class TrainedDeepNeuralNetwork(TrainedNeuralNetwork):
    pass


class UntrainedDeepNeuralNetwork(UntrainedNeuralNetwork):
    """``[LINEAR -> RELU] x (L-1) -> LINEAR -> SIGMOID`` network of arbitrary depth."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (20, 7, 5),
        seed: int | None = 1,
    ) -> None:
        self._hidden_layers = tuple(hidden_layers)
        self._seed = seed

    def fit(
        self,
        features: NDArray,
        outputs: NDArray,
        learning_rate: float = 0.0075,
        epoch: int = 2_500,
        print_cost: bool = False,
    ) -> TrainedDeepNeuralNetwork:
        trained = super().fit(features, outputs, learning_rate, epoch, print_cost)
        return TrainedDeepNeuralNetwork(trained.network, trained.costs)

    def _build_network(self, inputs: int, outputs: int) -> NeuralNetwork:
        rng = np.random.default_rng(self._seed)
        sizes = (inputs, *self._hidden_layers, outputs)
        output_layer = len(sizes) - 2

        layers: list[Layer] = [
            DenseLayer(
                sizes[index],
                sizes[index + 1],
                Sigmoid() if index == output_layer else ReLU(),
                rng,
            )
            for index in range(len(sizes) - 1)
        ]
        return NeuralNetwork(layers, BinaryCrossEntropy())
