import lmdb
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from preprocess import example_pb2


class DataSet(Dataset):
    def __init__(self, path_to_lmdb_dir, transform):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        with self._reader.begin() as txn:
            value = txn.get(self._keys[index])

        example = example_pb2.Example()
        example.ParseFromString(value)

        image = np.frombuffer(example.image, dtype=np.uint8)
        image = image.reshape([64, 64, 3])
        image = Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)

        num_digits = example.length
        digits = example.digits

        return image, num_digits, digits