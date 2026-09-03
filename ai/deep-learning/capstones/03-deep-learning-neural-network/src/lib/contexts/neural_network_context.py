from abc import abstractmethod
from collections.abc import Sequence

from numpy.typing import NDArray

from src.lib.abc.models import TrainedModel, UntrainedModel
from src.lib.managers.numpy_manager import ndarray_utils
from src.lib.nn import NeuralNetwork


class TrainedNeuralNetwork(TrainedModel):
    def __init__(self, network: NeuralNetwork, costs: Sequence[float] | None = None) -> None:
        self.network = network
        self.costs = list(costs) if costs is not None else []

    def predict_proba(self, features: NDArray) -> NDArray:
        return self.network.predict_proba(features)

    def predict(self, features: NDArray) -> NDArray:
        return self.predict_proba(features) > 0.5


class UntrainedNeuralNetwork(UntrainedModel):
    def fit(
        self,
        features: NDArray,
        outputs: NDArray,
        learning_rate: float = 0.0075,
        epoch: int = 2_500,
        print_cost: bool = False,
    ) -> TrainedNeuralNetwork:
        if not ndarray_utils.have_same_sample_count(features, outputs):
            raise ValueError("Number of samples in X and Y must match.")

        network = self._build_network(
            ndarray_utils.get_features_count(features),
            ndarray_utils.get_features_count(outputs),
        )
        costs = network.train(features, outputs, learning_rate, epoch, print_cost)
        return TrainedNeuralNetwork(network, costs)

    @abstractmethod
    def _build_network(self, inputs: int, outputs: int) -> NeuralNetwork:
        """Assemble the layer stack once the data has fixed the input/output sizes."""
