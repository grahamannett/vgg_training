from argparse import Namespace
import numpy as np
from tensorflow import keras

import argparse
import dataclasses
import os
import urllib.request
import sys
import tarfile

from pathlib import Path
from typing import List

from model import get_model


class P(argparse.ArgumentParser):
    DATASET_NAME = os.environ.get("DATASET_NAME", "imagenette2-160")
    DATASET_URL = os.environ.get("DATASET_URL", f"https://s3.amazonaws.com/fast-ai-imageclas/${DATASET_NAME}.tgz")
    OUTPUT = os.environ.get("OUTPUT", "/root/output/")

    EPOCHS: int = 20
    BATCH_SIZE: int = 32
    IMG_SIZE: List[int] = (256, 256)
    IMG_SHAPE: List[int] = IMG_SIZE + (3,)

    LABELS = {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute",
    }

    @classmethod
    def get_args(cls):
        args = cls()
        parser = argparse.ArgumentParser(description="run this thing")
        parser.add_argument("--EPOCHS", default=cls.EPOCHS)
        parser.add_argument("--OUTPUT", default=cls.OUTPUT)
        parser.parse_args(namespace=args)

        return args


def untar_data(filename: Path, outpath: Path = None) -> None:
    if outpath == None:
        outpath = filename.parent

    tar = tarfile.open(filename)
    tar.extractall(path=outpath)
    tar.close()


def fetch_data(url, filename=None):
    if filename is None:
        filename = Path(f"data/{url.split('/')[-1]}")
        filename.parent.mkdir(exist_ok=True)
    file, header = urllib.request.urlretrieve(url=url, filename=filename)
    return Path(file), header


def dataset_check(dataset_name):
    dataset_folder = Path("data") / dataset_name

    if dataset_folder.exists() is False:
        tar_file = dataset_folder.with_suffix(".tgz")
        if tar_file.exists() is False:
            tar_file, _ = fetch_data(url=P.DATASET_URL)
        untar_data(tar_file)

    return dataset_folder


def get_all_subfolder_images(folder, label_dict=P.LABELS, img_size: List[int] = P.IMG_SIZE):
    images = []
    labels = []
    for idx, key in enumerate(label_dict.keys()):
        label_folder = folder / key
        image_paths = list(label_folder.glob("*.JPEG"))
        imgs = [keras.preprocessing.image.load_img(p, target_size=img_size) for p in image_paths]
        # imgs = [keras.preprocessing.image.img_to_array(img) / 255 for img in imgs]
        imgs = [keras.preprocessing.image.img_to_array(img) for img in imgs]
        labels_ = [idx for _ in range(len(imgs))]
        images.extend(imgs)
        labels.extend(labels_)

    return np.array(images), keras.utils.to_categorical(np.array(labels))


if __name__ == "__main__":
    PARAMS = P.get_args()

    dataset_folder = dataset_check(PARAMS.DATASET_NAME)
    x_train, y_train = get_all_subfolder_images(dataset_folder / "train")
    x_val, y_val = get_all_subfolder_images(dataset_folder / "val")

    model = get_model(y_train.shape[1], full_size=True)
    history = model.fit(x_train, y_train, shuffle=True, epochs=PARAMS.EPOCHS)
    evaluate = model.evaluate(x_val, y_val)

    model.save(PARAMS.OUTPUT)