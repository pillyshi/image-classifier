from typing import Tuple, List, Optional, Dict

import numpy as np


class ImageClassifier:

    def __init__(self, window: Tuple[int, int], stride: Tuple[int, int], n_estimators: int, random_state: int, n_samples: Optional[int]):
        ...

    def train(self, images: List[np.ndarray], labels: List[str]):
        ...

    def classify(self, images: List[np.ndarray]) -> List[str]:
        ...

    def predict_proba(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        ...
