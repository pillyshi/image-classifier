from typing import Tuple, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ImageClassifier:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), n_estimators: int = 50, random_state: int = 1):
        self.window = window
        self.stride = stride
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, images: List[np.ndarray], labels: List[str]):
        X, y = self._fit_transform(images, labels)
        self.classifier.fit(X, y)
    
    def classify(self, images: List[np.ndarray]):
        X, ids = self._transform(images)
        proba = self.classifier.predict_proba(X)
        return [self.classifier.classes_[proba[ids == id].sum(axis=0).argmax()] for id in np.sort(np.unique(ids))]

    def _fit_transform(self, images: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        window = self.window
        stride = self.stride
        X: List[np.ndarray] = []
        y: List[str] = []
        for image, label in zip(images, labels):
            h, w = image.shape[0], image.shape[1]
            for i in range(window[0], h - window[0], stride[0]):
                for j in range(window[1], w - window[1], stride[1]):
                    x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                    X.append(x)
                    y.append(label)
        X = np.array(X) / 255
        y = np.array(y)
        return X, y

    def _transform(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        window = self.window
        stride = self.stride
        X: List[np.ndarray] = []
        ids: List[int] = []
        for id, image in enumerate(images):
            h, w = image.shape[0], image.shape[1]
            for i in range(window[0], h - window[0], stride[0]):
                for j in range(window[1], w - window[1], stride[1]):
                    x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                    X.append(x)
                    ids.append(id)
        X = np.array(X) / 255
        ids = np.array(ids)
        return X, ids
    