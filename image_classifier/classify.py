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
    in_model: str
    out_file: str


def main(option: Option) -> None:
    images: List[np.ndarray] = []
    filenames = glob.glob(os.path.join(option.in_dir, '*.*'))
    for filename in filenames:
        image = cv2.imread(filename)
        images.append(image)
    
    with open(option.in_model, 'rb') as fp:
        model: ImageClassifier = pickle.load(fp)
    
    labels = model.classify(images)
    with open(option.out_file, 'w') as fp:
        for filename, label in zip(filenames, labels):
            print(','.join([filename, label]), file=fp)


if __name__ == "__main__":
    option = Option().parse_args()
    main(option)
