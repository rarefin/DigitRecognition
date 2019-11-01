import lmdb
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from preprocess import example_pb2
import glob
import os
import json


def getRandomBoxes(file):
    with open(file, "r") as read_file:
        config = json.load(read_file)

    return config['sample_box_coordinate']


def getNeagativeSamplePaths(negative_example_dir):
    if negative_example_dir is None:
        return None
    else:
        return glob.glob(os.path.join(negative_example_dir, '*.jpg'))


def createNegativeSample(neg_sample_files, random_box_coordinates):
    rand_image_index = np.random.randint(low=0, high=len(neg_sample_files))
    rand_box_index = np.random.randint(low=0, high=len(random_box_coordinates))

    image = Image.open(neg_sample_files[rand_image_index])
    box = random_box_coordinates[rand_box_index]

    img_width, img_height = 512, 512

    box_width = box[2]-box[0]
    box_height = box[3]-box[1]

    box_width += 0.15 * box_width
    box_height += 0.15 * box_height

    x1 = np.random.randint(low=0, high=img_width-box_width)
    y1 = np.random.randint(low=0, high=img_height-box_height)

    image = image.crop([x1, y1, x1+box_width, y1+box_height])

    image = image.resize([64, 64])
    num_digits = 0
    digits = [10, 10, 10, 10, 10]
    return image, num_digits, digits


class DataSet(Dataset):
    def __init__(self, path_to_lmdb_dir, transform=None, negative_example_dir=None):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

        self.negative_example_dir = negative_example_dir
        self.neg_sample_files = getNeagativeSamplePaths(negative_example_dir)
        self.random_box_coordinates = getRandomBoxes("arbitray_box_sizes.json")

    def __len__(self):
        if self.negative_example_dir is None:
            return self._length
        else:
            return self._length+10000

    def __getitem__(self, index):
        if index < self._length:
            with self._reader.begin() as txn:
                value = txn.get(self._keys[index])

            example = example_pb2.Example()
            example.ParseFromString(value)

            image = np.frombuffer(example.image, dtype=np.uint8)
            image = image.reshape([64, 64, 3])
            image = Image.fromarray(image)

            num_digits = example.length
            digits = example.digits
        else:
            image, num_digits, digits = createNegativeSample(self.neg_sample_files, self.random_box_coordinates)

        if self._transform is not None:
            image = self._transform(image)

        return image, num_digits, digits