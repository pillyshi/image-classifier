from typing import Tuple, List, Union, Generator, Optional, Dict

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ImageTransformer:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), random_state: int = 1, n_samples: Optional[Tuple[int, int]] = None):
        self.window = window
        self.stride = stride
        self.random_state = random_state
        self.n_samples = n_samples

    def transform(self, image: np.ndarray) -> np.ndarray:
        rnd = np.random.RandomState(self.random_state)
        window = self.window
        stride = self.stride
        h, w = image.shape[0], image.shape[1]
        x_indices = np.arange(window[1], w - window[1], stride[1])
        y_indices = np.arange(window[0], h - window[0], stride[0])
        X: List[np.ndarray] = []
        if self.n_samples[0] is not None:
            y_indices = rnd.choice(y_indices, min(len(y_indices), self.n_samples[0]), replace=False)
        if self.n_samples[1] is not None:
            x_indices = rnd.choice(x_indices, min(len(x_indices), self.n_samples[1]), replace=False)
        for i in y_indices:
            for j in x_indices:
                x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                X.append(x)
        return np.array(X) / 255


class ImageClassifier:

    def __init__(self, window: Tuple[int, int] = (5, 5), stride: Tuple[int, int] = (5, 5), n_estimators: int = 50, random_state: int = 1, n_samples: Optional[Tuple[int, int]] = None):
        self.window = window
        self.stride = stride
        self.random_state = random_state
        self.n_samples = n_samples
        self.encoder = LabelEncoder()
        self.classifier = XGBClassifier(n_estimators=n_estimators, random_state=random_state, objective='multi:softmax', use_label_encoder=False)

    def train(self, images: List[np.ndarray], labels: List[Union[int, str]]):
        _labels = self.encoder.fit_transform(labels)
        X, y = self._fit_transform(images, _labels)
        self.classifier.fit(X, y)

    def classify(self, images: List[np.ndarray]) -> List[str]:
        # TODO
        return []

    def predict_proba(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        predicted = []
        for image in images:
            X = self._transform(image)
            proba = self.classifier.predict_proba(X).sum(axis=0).astype('float')
            proba /= proba.sum()
            labels = self.encoder.inverse_transform(np.arange(len(proba)))
            predicted.append({labels: p for labels, p in zip(labels, proba)})
        return predicted

    def _fit_transform(self, images: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for image, label in zip(images, labels):
            X = self._transform(image)
            Xs.append(X)
            ys.append([label for _ in range(len(X))])
        return np.vstack(Xs), np.concatenate(ys)

    def _transform(self, image: np.ndarray) -> np.ndarray:
        rnd = np.random.RandomState(self.random_state)
        window = self.window
        stride = self.stride
        h, w = image.shape[0], image.shape[1]
        x_indices = np.arange(window[1], w - window[1], stride[1])
        y_indices = np.arange(window[0], h - window[0], stride[0])
        X: List[np.ndarray] = []
        if self.n_samples[0] is not None:
            y_indices = rnd.choice(y_indices, min(len(y_indices), self.n_samples[0]), replace=False)
        if self.n_samples[1] is not None:
            x_indices = rnd.choice(x_indices, min(len(x_indices), self.n_samples[1]), replace=False)
        for i in y_indices:
            for j in x_indices:
                x = image[(i - window[0]):(i + window[0]), (j - window[1]):(j + window[1]), :].ravel()
                X.append(x)
        return np.array(X) / 255

    def _transform_multi(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
