from PIL import Image
from torch.utils.data import Dataset
from os.path import join
import numpy as np
import h5py
import glob


def selectRandomSample(BBInfo, n):
    length = len(BBInfo)
    while True:
        i = np.random.randint(low=0, high=length)

        item = BBInfo[i]
        num_digits = len(item['label'])

        if num_digits <= n:
            return i, item, num_digits


def getBoundingBoxAroundAllDigits(item):
    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x],
                                                           [item['left'], item['top'], item['width'],
                                                            item['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)

    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))

    return cropped_left, cropped_top, cropped_width, cropped_height


def getBBInfo(digit_struct_mat_file, length):
    boxes = []
    with h5py.File(digit_struct_mat_file, 'r') as file:
        for index in range(length):
            attrs = {}
            item = file['digitStruct']['bbox'][index].item()
            for key in ['label', 'left', 'top', 'width', 'height']:
                attr = file[item][key]
                values = [file[attr[i].item()][0][0] for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
                attrs[key] = values
            boxes.append(attrs)

    return boxes


class DataSet(Dataset):
    def __init__(self, data_dir, digit_struct_mat_file, transform=None):
        super(DataSet, self).__init__()
        self.data_dir = data_dir
        self.img_names = glob.glob(join(data_dir, '*.png'))
        self.BBInfo = getBBInfo(digit_struct_mat_file, len(self.img_names))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        item = self.BBInfo[index]
        num_digits = len(item['label'])

        if num_digits > 5:
            # number of digits are greater than n=5 in an image, we will select another image randomly
            index, item, num_digits = selectRandomSample(self.BBInfo, n=5)

        left, top, width, height = getBoundingBoxAroundAllDigits(item)

        path = join(self.data_dir, str(index+1) + '.png')
        img = Image.open(path)
        img = img.crop([left, top, left + width, top + height])
        img = img.resize([64, 64])

        if self.transform is not None:
            img = self.transform(img)

        label_of_digits = item['label']
        digits = [10, 10, 10, 10, 10]  # we look for maximum n=5 digits in an image, digit 10 represents no digit
        for i, label_of_digit in enumerate(label_of_digits):
            digits[i] = int(label_of_digit if label_of_digit != 10 else 0)


        return img, num_digits, digits