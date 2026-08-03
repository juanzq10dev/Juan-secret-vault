from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
IMAGE_SIZE = 64


class DatasetManager:
    def load_catvnoncat(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        train_x, train_y = self._load(DATASETS_DIR / "train_catvnoncat.h5", "train")
        test_x, test_y = self._load(DATASETS_DIR / "test_catvnoncat.h5", "test")
        classes = self._load_classes(DATASETS_DIR / "test_catvnoncat.h5")

        return (
            self._flatten_and_normalize(train_x),
            train_y,
            self._flatten_and_normalize(test_x),
            test_y,
            classes,
        )

    def _load(self, path: Path, prefix: str) -> tuple[NDArray, NDArray]:
        with h5py.File(path, "r") as f:
            x = np.array(f[f"{prefix}_set_x"])
            y = np.array(f[f"{prefix}_set_y"])
        return x, y

    def _load_classes(self, path: Path) -> NDArray:
        with h5py.File(path, "r") as f:
            return np.array(f["list_classes"])

    def _flatten_and_normalize(self, images: NDArray) -> NDArray:
        flattened = images.reshape(images.shape[0], -1)
        return flattened / 255.0

    def load_image(self, path: str) -> NDArray:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            pixels = np.array(resized)

        return pixels.reshape(-1) / 255.0


dataset_manager = DatasetManager()
