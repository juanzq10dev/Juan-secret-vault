from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.lib.managers.numpy_manager import ndarray_utils
from src.lib.nn.activation import Activation


class Layer(ABC):
    @abstractmethod
    def forward(self, A_prev: NDArray) -> NDArray:
        """Propagate activations forward, caching whatever ``backward`` needs."""

    @abstractmethod
    def backward(self, dA: NDArray) -> NDArray:
        """Consume the gradient w.r.t. this layer's output, store the parameter
        gradients, and return the gradient w.r.t. the previous layer's output."""

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        """Update the layer's parameters from the gradients stored by ``backward``."""


class DenseLayer(Layer):
    """Fully connected layer: ``A = activation(W @ A_prev + b)``.

    Weights use fan-in scaled initialisation (``randn / sqrt(inputs)``) to keep the
    signal variance stable across a deep stack.
    """

    def __init__(
        self,
        inputs: int,
        units: int,
        activation: Activation,
        rng: np.random.Generator,
    ) -> None:
        self._activation = activation
        self._W = ndarray_utils.randn(units, inputs, scale=1 / np.sqrt(inputs), rng=rng)
        self._b = ndarray_utils.zeros(units, 1)

        self._A_prev: NDArray | None = None
        self._Z: NDArray | None = None
        self._dW: NDArray | None = None
        self._db: NDArray | None = None

    def forward(self, A_prev: NDArray) -> NDArray:
        self._A_prev = A_prev
        self._Z = self._W @ A_prev + self._b
        return self._activation.apply(self._Z)

    def backward(self, dA: NDArray) -> NDArray:
        if self._A_prev is None or self._Z is None:
            raise RuntimeError("backward() called before forward()")

        m = self._A_prev.shape[1]
        dZ = dA * self._activation.derivative(self._Z)

        self._dW = (dZ @ self._A_prev.T) / m
        self._db = np.sum(dZ, axis=1, keepdims=True) / m

        return self._W.T @ dZ

    def apply_gradients(self, learning_rate: float) -> None:
        if self._dW is None or self._db is None:
            raise RuntimeError("apply_gradients() called before backward()")

        self._W = self._W - learning_rate * self._dW
        self._b = self._b - learning_rate * self._db
