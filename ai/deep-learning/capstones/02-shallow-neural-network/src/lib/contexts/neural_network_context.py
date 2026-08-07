from numpy.typing import NDArray

from src.lib.abc.model_strategy import ModelStrategy
from src.lib.abc.models import TrainedModel, UntrainedModel
from src.lib.managers.numpy_manager import ndarray_utils
from src.lib.types.network import LayerSizes, Parameters


class TrainedNeuralNetwork(TrainedModel):
    def __init__(self, parameters: Parameters, strategy: ModelStrategy) -> None:
        self.parameters = parameters
        self.strategy = strategy

    def predict_proba(self, features: NDArray) -> NDArray:
        cache = self.strategy.forward_propagation(features, self.parameters)
        return cache.A2

    def predict(self, features: NDArray) -> NDArray:
        return self.predict_proba(features) > 0.5


class UntrainedNeuralNetwork(UntrainedModel):
    def __init__(self, strategy: ModelStrategy, hidden_units: int) -> None:
        self.strategy = strategy
        self.hidden_units = hidden_units

    def fit(
        self,
        features: NDArray,
        outputs: NDArray,
        learning_rate: float = 1.2,
        epoch: int = 10_000,
        print_cost: bool = False,
    ) -> TrainedNeuralNetwork:
        if not ndarray_utils.have_same_sample_count(features, outputs):
            raise ValueError("Number of samples in X and Y must match.")

        sizes = self._layer_sizes(features, outputs)
        parameters = self.strategy.initialize_parameters(sizes)
        parameters = self.strategy.gradient_descent(
            features, outputs, parameters, learning_rate, epoch, print_cost
        )
        return TrainedNeuralNetwork(parameters, self.strategy)

    def _layer_sizes(self, X: NDArray, Y: NDArray) -> LayerSizes:
        return LayerSizes(
            n_x=ndarray_utils.get_features_count(X),
            n_h=self.hidden_units,
            n_y=ndarray_utils.get_features_count(Y),
        )
