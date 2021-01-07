import os
import glob
import pickle
from typing import List
from dataclasses import dataclass

import cv2
import numpy as np
from tap import Tap

from .model import ImageClassifier


class Option(Tap):
    in_dir: str
    in_model: str
    out_file: str


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


def main() -> None:
    option = Option().parse_args()

    images: List[np.ndarray] = []
    filenames = glob.glob(os.path.join(option.in_dir, '*.*'))
    for filename in filenames:
        image = cv2.imread(filename)
        images.append(image)

    model = load_model(option.in_model)
    labels = model.classify(images)

    rows = [Row(filename, label) for filename, label in zip(filenames, labels)]
    to_csv(rows, option.out_file)


if __name__ == "__main__":
    main(option)
