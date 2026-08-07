from abc import ABC, abstractmethod

from numpy.typing import NDArray

from src.lib.types.network import ForwardCache, Gradients, LayerSizes, Parameters


class ModelStrategy(ABC):
    @abstractmethod
    def initialize_parameters(self, sizes: LayerSizes) -> Parameters:
        pass

    @abstractmethod
    def forward_propagation(self, X: NDArray, parameters: Parameters) -> ForwardCache:
        pass

    @abstractmethod
    def cost_function(self, A2: NDArray, Y: NDArray) -> float:
        pass

    @abstractmethod
    def backward_propagation(
        self,
        parameters: Parameters,
        cache: ForwardCache,
        X: NDArray,
        Y: NDArray,
    ) -> Gradients:
        pass

    @abstractmethod
    def update_parameters(
        self,
        parameters: Parameters,
        gradients: Gradients,
        learning_rate: float,
    ) -> Parameters:
        pass

    @abstractmethod
    def gradient_descent(
        self,
        X: NDArray,
        Y: NDArray,
        parameters: Parameters,
        learning_rate: float,
        epoch: int,
        print_cost: bool = False,
    ) -> Parameters:
        pass
