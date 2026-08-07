import numpy as np
from numpy.typing import NDArray

from src.lib.abc.model_strategy import ModelStrategy
from src.lib.managers.numpy_manager import ndarray_utils
from src.lib.types.network import ForwardCache, Gradients, LayerSizes, Parameters


class ShallowNeuralNetworkStrategy(ModelStrategy):
    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    def initialize_parameters(self, sizes: LayerSizes) -> Parameters:
        rng = np.random.default_rng(self._seed)

        W1 = ndarray_utils.randn(sizes.n_h, sizes.n_x, scale=0.01, rng=rng)
        b1 = ndarray_utils.zeros(sizes.n_h, 1)
        W2 = ndarray_utils.randn(sizes.n_y, sizes.n_h, scale=0.01, rng=rng)
        b2 = ndarray_utils.zeros(sizes.n_y, 1)

        return Parameters(W1=W1, b1=b1, W2=W2, b2=b2)

    def forward_propagation(self, X: NDArray, parameters: Parameters) -> ForwardCache:
        Z1 = parameters.W1 @ X + parameters.b1
        A1 = self._tanh(Z1)
        Z2 = parameters.W2 @ A1 + parameters.b2
        A2 = self._sigmoid(Z2)

        return ForwardCache(Z1=Z1, A1=A1, Z2=Z2, A2=A2)

    def cost_function(self, A2: NDArray, Y: NDArray) -> float:
        m = ndarray_utils.get_samples_count(Y)
        return float(-np.sum(self._cross_entropy(A2, Y)) / m)

    def backward_propagation(
        self,
        parameters: Parameters,
        cache: ForwardCache,
        X: NDArray,
        Y: NDArray,
    ) -> Gradients:
        m = ndarray_utils.get_samples_count(X)

        dZ2 = cache.A2 - Y
        dW2 = (dZ2 @ cache.A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (parameters.W2.T @ dZ2) * self._tanh_derivative(cache.A1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return Gradients(dW1=dW1, db1=db1, dW2=dW2, db2=db2)

    def update_parameters(
        self,
        parameters: Parameters,
        gradients: Gradients,
        learning_rate: float,
    ) -> Parameters:
        return Parameters(
            W1=parameters.W1 - learning_rate * gradients.dW1,
            b1=parameters.b1 - learning_rate * gradients.db1,
            W2=parameters.W2 - learning_rate * gradients.dW2,
            b2=parameters.b2 - learning_rate * gradients.db2,
        )

    def gradient_descent(
        self,
        X: NDArray,
        Y: NDArray,
        parameters: Parameters,
        learning_rate: float,
        epoch: int,
        print_cost: bool = False,
    ) -> Parameters:
        result_parameters = parameters

        for i in range(epoch):
            cache = self.forward_propagation(X, result_parameters)
            cost = self.cost_function(cache.A2, Y)
            gradients = self.backward_propagation(result_parameters, cache, X, Y)
            result_parameters = self.update_parameters(result_parameters, gradients, learning_rate)

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")

        return result_parameters

    def _sigmoid(self, Z: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-Z))

    def _tanh(self, Z: NDArray) -> NDArray:
        return np.tanh(Z)

    def _tanh_derivative(self, A1: NDArray) -> NDArray:
        return 1 - A1**2

    def _cross_entropy(self, A2: NDArray, Y: NDArray) -> NDArray:
        return Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
