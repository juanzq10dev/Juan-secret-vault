from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class _PredictsProbability(Protocol):
    def predict_proba(self, features: NDArray) -> NDArray: ...


class PlotManager:
    def scatter(self, X: NDArray, Y: NDArray, title: str) -> None:
        plt.figure()
        plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(title)

    def decision_boundary(
        self, model: _PredictsProbability, X: NDArray, Y: NDArray, title: str
    ) -> None:
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()].T

        Z = model.predict_proba(grid).reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors="k")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(title)

    def save(self, path: str | Path) -> None:
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        plt.show()


plot_manager = PlotManager()
