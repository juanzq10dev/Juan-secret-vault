# Planar Data Classification with One Hidden Layer

Object-oriented port of the Coursera "Planar data classification with one hidden layer"
assignment. See [`Spec.md`](./Spec.md) for the full design.

## Usage

```bash
uv sync
uv run main.py                            # planar flower, n_h = 4
uv run main.py --hidden-units 5           # tune hidden layer size
uv run main.py --dataset noisy_moons      # section 7 of the assignment
uv run main.py --sweep                    # n_h ∈ {1,2,3,4,5}, accuracy table
```

Decision-boundary plots are written to `plots/`.

## Dev

```bash
uv pip install -r requirements-dev.txt
uv run mypy src main.py
uv run ruff check src main.py
```
