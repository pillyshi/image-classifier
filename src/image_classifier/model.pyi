from typing import Tuple, List

import numpy as np


class ImageClassifier:

    def __init__(self, window: Tuple[int, int], stride: Tuple[int, int], n_estimators: int, random_state: int):
        ...

    def train(self, images: List[np.ndarray], labels: List[str]):
        ...

    def classify(self, images: List[np.ndarray]) -> List[str]:
        ...
