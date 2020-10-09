import os.path as op

import numpy as np
import cv2
from torch.utils import data
from PIL import Image

import base_settings
from utils import fs_utils


class MiniImageNet(data.Dataset):

    def __init__(self, train, transform=None, corrupted_labels_path=None):
        if train:
            self.ds_pickle = fs_utils.read_pickle(op.join(base_settings.MINI_IMAGENET_ROOT, 'train.pickle'))
        else:
            self.ds_pickle = fs_utils.read_pickle(op.join(base_settings.MINI_IMAGENET_ROOT, 'test.pickle'))

        if not train and corrupted_labels_path is not None:
            raise ValueError("Don't pass corrupted_labels_path for dataset in test mode")

        if corrupted_labels_path is not None:
            self.ds_pickle['labels'] = np.array(fs_utils.read_json(corrupted_labels_path), dtype=np.long)

        self.transform = transform

    def __len__(self):
        return len(self.ds_pickle['labels'])

    def __getitem__(self, idx):
        img = self.ds_pickle['data'][idx, :, :, :]
        label = self.ds_pickle['labels'][idx]

        img = cv2.resize(img, (42, 42), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = img.transpose((2, 0, 1))

        return img, label
