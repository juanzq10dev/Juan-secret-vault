# Building a Deep Neural Network: Step by Step

Object-oriented port of the Coursera "Building your Deep Neural Network: Step by Step"
assignment (Course 1, Week 4), applied to the cat vs non-cat dataset. An arbitrary-depth
`[LINEAR -> RELU] x (L-1) -> LINEAR -> SIGMOID` network trained with batch gradient descent.

## Architecture

The reusable building blocks live in `src/lib/nn/` (one file per concept, each holding
its abstraction plus the concrete implementations):

| `src/lib/nn/` | Responsibility |
| --- | --- |
| `layer.py` — `Layer`, `DenseLayer` | a layer owns `W`, `b`, an `Activation`; `forward` / `backward` / `apply_gradients` |
| `activation.py` — `Activation`, `ReLU`, `Sigmoid` | `apply` + `derivative` |
| `loss.py` — `Loss`, `BinaryCrossEntropy` | `cost` + `gradient` (the `dAL` seed for backprop) |
| `network.py` — `NeuralNetwork` | ordered stack of layers + a loss; runs the forward chain and the training loop |

`src/neural_network_models/deep_neural_network.py` wraps that in the `fit` / `predict`
API (`Untrained`/`TrainedDeepNeuralNetwork`); the untrained side builds the layer stack
once the data fixes the input size.

A shallow net would just be a shorter stack, so there is no separate "shallow" vs "deep"
type — only a different list of layers.

## Usage

```bash
uv sync
uv run main.py                                   # 4-layer net [12288, 20, 7, 5, 1], 2500 epochs
uv run main.py --hidden-layers 7                 # 2-layer net
uv run main.py --hidden-layers 20 7 5 --epochs 1500
uv run main.py --learning-rate 0.01 --quiet
```

The learning curve is written to `plots/`. The default 4-layer net reaches ~100% train /
~78% test accuracy.

Weights use fan-in scaled initialisation (`randn / sqrt(n_prev)`) rather than the
notebook's fixed `* 0.01`; on the 12288-dimensional image input the fixed scale vanishes
the gradients and the network never leaves the majority-class baseline.

## Dev

```bash
uv pip install -r requirements-dev.txt
uv run mypy src main.py
uv run ruff check src main.py
```
