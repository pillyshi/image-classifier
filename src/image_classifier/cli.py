import os
import glob
import pickle
import json
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import xgboost as xgb
from tap import Tap
from sklearn.preprocessing import LabelEncoder

from .model import ImageTransformer, ImageClassifier


class TrainOption(Tap):
    name: str = 'train'
    in_dir: str
    out_model: str
    window: Tuple[int, int] = (5, 5)
    stride: Tuple[int, int] = (5, 5)
    n_estimators: int = 50
    random_state: int = 1
    n_samples: Tuple[int, int] = (None, None)


class TrainXGBoostOption(Tap):
    name: str = 'train_xgboost'
    filename: str
    out_model: str
    n_estimators: int = 50
    random_state: int = 1


class ClassifyOption(Tap):
    name: str = 'classify'
    filename: str
    in_model: str


class ClassifyXGBoostOption(Tap):
    name: str = 'classify_xgboost'
    filename: str
    in_model: str
    in_labels: str
    window: Tuple[int, int] = (5, 5)
    stride: Tuple[int, int] = (5, 5)
    n_samples: Tuple[int, int] = (None, None)
    random_state: int = 1


class CropOption(Tap):
    name: str = 'crop'
    filename: str
    x: int = 0
    y: int = 0
    w: Optional[int] = None
    h: Optional[int] = None
    out_file: Optional[str] = None


class DumpOption(Tap):
    name: str = 'dump'
    in_dir: str
    out_dir: str
    window: Tuple[int, int] = (5, 5)
    stride: Tuple[int, int] = (5, 5)
    n_samples: Tuple[int, int] = (None, None)
    random_state: int = 1


@dataclass
class Row:
    filename: str
    label: str


def load_model(filename: str) -> ImageClassifier:
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def to_csv(rows: List[Row], filename: str):
    with open(filename, 'w') as fp:
        for row in rows:
            line = f'{row.filename},{row.label}'
            fp.write(line + '\n')


def train(option: TrainOption) -> None:
    images: List[np.ndarray] = []
    labels: List[str] = []
    for label in os.listdir(option.in_dir):
        for filename in glob.glob(os.path.join(option.in_dir, label, '*')):
            image = cv2.imread(filename)
            images.append(image)
            labels.append(label)
    classifier = ImageClassifier(option.window, option.stride, option.n_estimators, option.random_state, option.n_samples)
    classifier.train(images, labels)
    with open(option.out_model, 'wb') as fp:
        pickle.dump(classifier, fp)


def train_xgboost(option: TrainXGBoostOption):
    data = xgb.DMatrix(option.filename)
    # data.set_label(data.get_label() - 1)
    print(np.unique(data.get_label()))
    print(len(np.unique(data.get_label())))
    params = {
        'objective': 'multi:softprob',
        'random_state': option.random_state,
        'num_class': len(np.unique(data.get_label()))
    }
    bst = xgb.train(params, data, option.n_estimators)
    with open(option.out_model, 'wb') as fp:
        pickle.dump(bst, fp)


def classify(option: ClassifyOption) -> None:
    image: np.ndarray = cv2.imread(option.filename)

    model = load_model(option.in_model)
    proba = model.predict_proba([image])
    print(json.dumps(proba, indent=2))


def classify_xgboost(option: ClassifyXGBoostOption):
    with open(option.in_model, 'rb') as fp:
        bst = pickle.load(fp)
    with open(option.in_labels) as fp:
        labels = json.load(fp)
    transformer = ImageTransformer(
        window=option.window,
        stride=option.stride,
        random_state=option.random_state,
        n_samples=option.n_samples
    )


    image: np.ndarray = cv2.imread(option.filename)
    X = transformer.transform(image)

    data = xgb.DMatrix(X)
    proba = bst.predict(data)
    proba = proba.sum(axis=0) / proba.sum()
    print(json.dumps({l: p for l, p in zip(labels, proba.astype('float'))}, indent=2))


def crop(option: CropOption) -> None:
    image = cv2.imread(option.filename)
    x = option.x
    y = option.y
    w = image.shape[1] if option.w is None else option.w
    h = image.shape[0] if option.h is None else option.h
    cropped = image[y:y + h, x:x + w, :]
    if option.out_file is not None:
        cv2.imwrite(option.out_file, cropped)
    else:
        cv2.imshow(option.filename, cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def dump(option: DumpOption) -> None:
    transformer = ImageTransformer(
        window=option.window,
        stride=option.stride,
        random_state=option.random_state,
        n_samples=option.n_samples
    )

    labels = [name.split('/')[-1] for name in glob.glob(os.path.join(option.in_dir, '*')) if os.path.isdir(name)]

    fp = open(os.path.join(option.out_dir, 'data.txt'), 'w')
    for class_, label in enumerate(labels):
        for filename in glob.glob(os.path.join(option.in_dir, label, '*')):
            image = cv2.imread(filename)
            X = transformer.transform(image)
            for x in X:
                line = f'{class_} ' + ' '.join(f'{i}:{x_i}' for i, x_i in enumerate(x))
                print(line, file=fp)
    fp.close()
    with open(os.path.join(option.out_dir, 'labels.json'), 'w') as fp:
        json.dump(labels, fp)


class Option(Tap):
    name: str = ''

    def configure(self):
        self.add_subparsers(help='')
        self.add_subparser('train', TrainOption, help='')
        self.add_subparser('train_xgboost', TrainXGBoostOption, help='')
        self.add_subparser('classify', ClassifyOption, help='')
        self.add_subparser('classify_xgboost', ClassifyXGBoostOption, help='')
        self.add_subparser('crop', CropOption, help='')
        self.add_subparser('dump', DumpOption, help='')


def main() -> None:
    option = Option().parse_args()
    eval(option.name)(option)
