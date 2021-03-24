from typing import Tuple, List, Optional, Dict

import numpy as np


class ImageTransformer:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), random_state: int = 1, n_samples: Optional[Tuple[int, int]] = None):
        ...

    def transform(self, image: np.ndarray) -> np.ndarray:
        ...


class ImageClassifier:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), n_estimators: int = 50, random_state: int = 1, n_samples: Optional[Tuple[int, int]] = None):
        ...

    def train(self, images: List[np.ndarray], labels: List[str]):
        ...

    def classify(self, images: List[np.ndarray]) -> List[str]:
        ...

    def predict_proba(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        ...
