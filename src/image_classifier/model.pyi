from typing import List

import numpy as np


class ImageClassifier:

    def train(self, images: List[np.ndarray], labels: List[str]): ...

    def classify(self, images: List[np.ndarray]) -> List[str]: ...
