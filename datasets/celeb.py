import os.path as op

import cv2
import tensorflow as tf
import numpy as np

from utils import img_ops
from utils.logger import log
from utils import fs_utils


@log
class CelebDataset(object):
    __scale_img_choices__ = {'0_to_1', '-1_to_1', None}

    def __init__(self, celeb_images, batch_size, crop_size, out_size, scale_img, read_precomputed=True):
        assert scale_img in self.__scale_img_choices__

        self.crop_size = crop_size
        self.out_size = out_size
        self.scale_img = scale_img
        self.celeb_images = celeb_images

        self.batch_size = batch_size

        self.celeb_files_list_test, self.celeb_files_list_train = None, None
        self.celeb_train_arr, self.celeb_test_arr = None, None

        if read_precomputed:
            if op.exists(op.join(celeb_images, 'train.npy')):
                self.logger.info("Dataset exists in precomputed numpy array form, reading from them ...")
                self.celeb_train_arr = np.load(op.join(celeb_images, 'train.npy'))
                self.celeb_test_arr = np.load(op.join(celeb_images, 'test.npy'))
                self.train_size, self.test_size = self.celeb_train_arr.shape[0], self.celeb_test_arr.shape[0]
            else:
                self.celeb_files_list_train = fs_utils.read_json(op.join(celeb_images, 'train.json'))
                self.celeb_files_list_test = fs_utils.read_json(op.join(celeb_images, 'test.json'))
                self.train_size, self.test_size = len(self.celeb_files_list), len(self.celeb_files_list_test)
        else:
            raise ValueError("Currently, read_precomputed must be True")

        self.logger.info("Celeb faces train/test dataset sizes = (%d, %d)" % (self.train_size, self.test_size))

    def settings(self):
        return {'ds_path': self.celeb_images, 'crop_size': self.crop_size, 'image_size': self.out_size,
                'scale_img': self.scale_img, 'batch_size': self.batch_size}

    def generate_train_mb(self, fill_to_batch_size=True):
        if self.celeb_train_arr is None:
            return self.generate_from_img_list(self.celeb_files_list, fill_to_batch_size)
        else:
            return self.generate_from_npy_arr(self.celeb_train_arr, fill_to_batch_size)

    def generate_test_mb(self, fill_to_batch_size=True):
        if self.celeb_test_arr is None:
            assert self.celeb_files_list_test is not None
            return self.generate_from_img_list(self.celeb_files_list_test, fill_to_batch_size)
        else:
            return self.generate_from_npy_arr(self.celeb_test_arr, fill_to_batch_size)

    def generate_from_npy_arr(self, npy_arr, fill_to_batch_size):
        for i in range(0, npy_arr.shape[0], self.batch_size):
            images_batch = [img for img in npy_arr[i: i + self.batch_size]]
            if len(images_batch) < self.batch_size and fill_to_batch_size:
                images_batch += [img for img in npy_arr[0: self.batch_size - len(images_batch)]]

            result = [self.transform(img) for img in images_batch] if self.scale_img else images_batch
            yield np.array(result, dtype=np.float32)

    def generate_from_img_list(self, img_list, fill_to_batch_size):
        for i in range(0, len(img_list), self.batch_size):
            batch_files_list = img_list[i: i + self.batch_size]
            if len(batch_files_list) < self.batch_size and fill_to_batch_size:
                batch_files_list += img_list[0: self.batch_size - len(batch_files_list)]
            images_batch = [self.read_img(p) for p in batch_files_list]
            yield np.array(images_batch, dtype=np.float32)

    def read_img(self, img_path):
        img = cv2.imread(img_path)
        crop = img_ops.center_crop(img, self.crop_size, self.crop_size) if self.crop_size is not None else img
        resized_img = cv2.resize(crop, (self.out_size, self.out_size)) if self.out_size is not None else crop
        return self.transform(resized_img) if self.scale_img else resized_img

    def img_size(self):
        return self.out_size

    def train_ds_size(self):
        return self.train_size

    def test_ds_size(self):
        return self.test_size

    def transform(self, img_arr):
        return transform(img_arr, self.scale_img)

    def inv_transform(self, transformed_img):
        return inv_transform(transformed_img, self.scale_img)

    @staticmethod
    def transform_tf(t_ph):
        return tf.image.convert_image_dtype(t_ph, dtype=tf.float32, saturate=True)

    @staticmethod
    def inv_transform_tf(t_ph):
        return tf.image.convert_image_dtype(t_ph, dtype=tf.uint8, saturate=True)


def transform(img_arr, scale_img_opt):
    if scale_img_opt == '0_to_1':
        return img_ops.bound_image_values_01(img_arr).astype(np.float32)
    else:
        return img_ops.bound_image_values(img_arr).astype(np.float32)


def inv_transform(transformed_img, scale_img_opt):
    if scale_img_opt == '0_to_1':
        return img_ops.unbound_images_values_01(transformed_img).astype(np.uint8)
    else:
        return img_ops.unbound_image_values(transformed_img).astype(np.uint8)


def get_celeb_ds_from_train_config(train_config):
    if train_config['ds']['scale_img'] is True:
        scale_img_opt = '0_to_1'
    elif train_config['ds']['scale_img'] is not None:
        scale_img_opt = train_config['ds']['scale_img']
    else:
        scale_img_opt = None

    batch_size = train_config['ds']['batch_size'] if 'batch_size' in train_config['ds'] else 64

    celeb_ds = CelebDataset(train_config['ds']['ds_path'], batch_size,
                            train_config['ds']['crop_size'], train_config['ds']['image_size'],
                            scale_img_opt, read_precomputed=True)
    return celeb_ds
