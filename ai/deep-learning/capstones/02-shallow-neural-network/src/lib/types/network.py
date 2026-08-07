from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True)
class LayerSizes:
    n_x: int
    n_h: int
    n_y: int


@dataclass(frozen=True)
class Parameters:
    W1: NDArray
    b1: NDArray
    W2: NDArray
    b2: NDArray


@dataclass(frozen=True)
class ForwardCache:
    Z1: NDArray
    A1: NDArray
    Z2: NDArray
    A2: NDArray


@dataclass(frozen=True)
class Gradients:
    dW1: NDArray
    db1: NDArray
    dW2: NDArray
    db2: NDArray
