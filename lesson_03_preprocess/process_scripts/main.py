import pickle
from typing import List

import argparse
import gzip
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_PATH = pathlib.Path("/opt/ml/processing")


def training_images(path: pathlib.Path) -> np.ndarray:
    with gzip.open(path, "r") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), "big")
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), "big")
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), "big")
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (image_count, row_count, column_count)
        )
        return images


def training_labels(path: pathlib.Path) -> np.ndarray:
    with gzip.open(path, "r") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), "big")
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def save_numpy_array(path: pathlib.Path, arrays: List[np.array]):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        pickle.dump(arrays, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    images = training_images(ROOT_PATH / "input/train-images-idx3-ubyte.gz")
    labels = training_labels(ROOT_PATH / "input/train-labels-idx1-ubyte.gz")

    train_images, valid_images, train_y, valid_y = train_test_split(
        images, labels, test_size=args.test_size
    )

    save_numpy_array(ROOT_PATH / "train/train.npy", [train_images, train_y])
    save_numpy_array(ROOT_PATH / "valid/valid.npy", [valid_images, valid_y])
