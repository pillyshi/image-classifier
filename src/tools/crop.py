import cv2
from tap import Tap


class Option(Tap):
    filename: str
    x: int = 0
    y: int = 0
    w: int = None
    h: int = None
    out_file: str = None


def main() -> None:
    option = Option().parse_args()
    image = cv2.imread(option.filename)
    x = option.x
    y = option.y
    w = image.shape[1] if option.w is None else option.w
    h = image.shape[0] if option.h is None else option.h
    cropped = image[y:y+h, x:x+w, :]
    if option.out_file is not None:
        cv2.imwrite(option.out_file, cropped)
    else:
        cv2.imshow(option.filename, cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
