from collections.abc import Sequence

from numpy.typing import NDArray

from src.lib.nn.layer import Layer
from src.lib.nn.loss import Loss

COST_RECORD_INTERVAL = 100


class NeuralNetwork:
    """An ordered stack of layers trained end to end against a loss."""

    def __init__(self, layers: Sequence[Layer], loss: Loss) -> None:
        self._layers = layers
        self._loss = loss

    def predict_proba(self, X: NDArray) -> NDArray:
        activations = X
        for layer in self._layers:
            activations = layer.forward(activations)
        return activations

    def train(
        self,
        X: NDArray,
        Y: NDArray,
        learning_rate: float,
        epochs: int,
        print_cost: bool = False,
    ) -> list[float]:
        costs: list[float] = []

        for epoch in range(epochs):
            AL = self.predict_proba(X)
            cost = self._loss.cost(AL, Y)

            gradient = self._loss.gradient(AL, Y)
            for layer in reversed(self._layers):
                gradient = layer.backward(gradient)

            for layer in self._layers:
                layer.apply_gradients(learning_rate)

            if epoch % COST_RECORD_INTERVAL == 0:
                costs.append(cost)
                if print_cost:
                    print(f"Cost after iteration {epoch}: {cost:.6f}")

        return costs
