from typing import Tuple, List, Union, Generator, Optional, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ImageClassifier:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), n_estimators: int = 50, random_state: int = 1, n_samples: Optional[int] = None):
        self.window = window
        self.stride = stride
        self.random_state = random_state
        self.n_samples = n_samples
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, images: List[np.ndarray], labels: List[Union[int, str]]):
        X, y = self._fit_transform(images, labels)
        self.classifier.fit(X, y)

    def classify(self, images: List[np.ndarray]) -> List[str]:
        proba = []
        ids_all = []
        for X, ids in self._transform(images):
            proba.append(self.classifier.predict_proba(X))
            ids_all = np.concatenate([ids_all, ids])
        proba_all = np.vstack(proba)
        return [self.classifier.classes_[proba_all[ids == id].sum(axis=0).argmax()] for id in np.sort(np.unique(ids_all))]

    def predict_proba(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        X, ids = self._transform(images)
        proba = self.classifier.predict_proba(X)
        predicted = []
        for id_ in np.sort(np.unique(ids)):
            predicted.append({c: p for c, p in zip(self.classifier.classes_, proba[ids == id_].sum(axis=0) / proba[ids == id_].sum())})
        return predicted

    def _fit_transform(self, images: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        rnd = np.random.RandomState(self.random_state)
        window = self.window
        stride = self.stride
        X: List[np.ndarray] = []
        y: List[str] = []
        for image, label in zip(images, labels):
            h, w = image.shape[0], image.shape[1]
            x_indices = np.arange(window[1], w - window[1], stride[1])
            y_indices = np.arange(window[0], h - window[0], stride[0])
            if self.n_samples is not None:
                x_indices = rnd.choice(x_indices, self.n_samples)
                y_indices = rnd.choice(y_indices, self.n_samples)
            for i in y_indices:
                for j in x_indices:
                    x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                    X.append(x)
                    y.append(label)
        X = np.array(X) / 255
        y = np.array(y)
        return X, y

    def _transform(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        rnd = np.random.RandomState(self.random_state)
        window = self.window
        stride = self.stride
        X: List[np.ndarray] = []
        ids: List[int] = []
        for id, image in enumerate(images):
            h, w = image.shape[0], image.shape[1]
            x_indices = np.arange(window[1], w - window[1], stride[1])
            y_indices = np.arange(window[0], h - window[0], stride[0])
            if self.n_samples is not None:
                x_indices = rnd.choice(x_indices, self.n_samples)
                y_indices = rnd.choice(y_indices, self.n_samples)
            for i in y_indices:
                for j in x_indices:
                    x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                    X.append(x)
                    ids.append(id)
        X = np.array(X) / 255
        ids = np.array(ids)
        return X, ids

    def _iter_transform(self, images: List[np.ndarray], batch_size=100) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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
                    if len(X) >= batch_size:
                        yield np.array(X) / 255, np.array(ids)
                        X = []
                        ids = []
