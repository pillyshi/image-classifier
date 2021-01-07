import os
import glob
import pickle
from typing import List

import cv2
import numpy as np
from tap import Tap

from .model import ImageClassifier


class Option(Tap):
    in_dir: str
    out_model: str


def main() -> None:
    option = Option().parse_args()

    images: List[np.ndarray] = []
    labels: List[str] = []
    for label in os.listdir(option.in_dir):
        for filename in glob.glob(os.path.join(option.in_dir, label, '*')):
            image = cv2.imread(filename)
            images.append(image)
            labels.append(label)
    classifier = ImageClassifier()
    classifier.train(images, labels)
    with open(option.out_model, 'wb') as fp:
        pickle.dump(classifier, fp)


if __name__ == "__main__":
    main(option)
