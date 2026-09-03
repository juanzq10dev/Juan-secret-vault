from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

COST_RECORD_INTERVAL = 100


class PlotManager:
    def cost_curve(
        self,
        costs: Sequence[float],
        learning_rate: float,
        interval: int = COST_RECORD_INTERVAL,
    ) -> None:
        plt.figure()
        plt.plot([i * interval for i in range(len(costs))], costs)
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.title(f"Learning curve (learning rate = {learning_rate})")

    def save(self, path: str | Path) -> None:
        plt.savefig(path)
        plt.close()

    def show(self) -> None:
        plt.show()


plot_manager = PlotManager()
