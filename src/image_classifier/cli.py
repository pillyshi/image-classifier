import os
import glob
import pickle
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from tap import Tap

from .model import ImageClassifier


class TrainOption(Tap):
    name: str = 'train'
    in_dir: str
    out_model: str
    window: Tuple[int, int] = (5, 5)
    stride: Tuple[int, int] = (5, 5)
    n_estimators: int = 50
    random_state: int = 1


class ClassifyOption(Tap):
    name: str = 'classify'
    filename: str
    in_model: str


class CropOption(Tap):
    name: str = 'crop'
    filename: str
    x: int = 0
    y: int = 0
    w: Optional[int] = None
    h: Optional[int] = None
    out_file: Optional[str] = None


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
    classifier = ImageClassifier(option.window, option.stride, option.n_estimators, option.random_state)
    classifier.train(images, labels)
    with open(option.out_model, 'wb') as fp:
        pickle.dump(classifier, fp)


def classify(option: ClassifyOption) -> None:
    image: np.ndarray = cv2.imread(option.filename)

    model = load_model(option.in_model)
    labels = model.classify([image])

    print(labels[0])


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


class Option(Tap):
    name: str = ''

    def configure(self):
        self.add_subparsers(help='')
        self.add_subparser('train', TrainOption, help='')
        self.add_subparser('classify', ClassifyOption, help='')
        self.add_subparser('crop', CropOption, help='')


def main() -> None:
    option = Option().parse_args()
    eval(option.name)(option)
