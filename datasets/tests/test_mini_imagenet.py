import unittest

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import base_settings
from datasets import mini_imagenet
from src.transforms import get_train_transform


class MiniImageNetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 128
        cls.dest_batch_size = (cls.batch_size, 3, 42, 42)

    def dl_create_func(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=1,
                          shuffle=True)

    def test_train_mode(self):
        train_ds = mini_imagenet.MiniImageNet(train=True, transform=None)
        train_dl = iter(self.dl_create_func(train_ds))

        x, y = next(train_dl)
        assert len(x.shape) == 4
        assert x.shape == self.dest_batch_size
        assert y.shape == (self.batch_size,)

    def test_test_mode(self):
        train_ds = mini_imagenet.MiniImageNet(train=False, transform=None)
        test_dl = iter(self.dl_create_func(train_ds))

        x, y = next(test_dl)
        assert len(x.shape) == 4
        assert x.shape == self.dest_batch_size
        assert y.shape == (self.batch_size,)

    def test_train_mode_with_transform(self):
        train_ds = mini_imagenet.MiniImageNet(train=True, transform=get_train_transform(True, 'imagenet'))
        train_dl = iter(self.dl_create_func(train_ds))

        x, y = next(train_dl)
        assert len(x.shape) == 4
        assert x.shape == self.dest_batch_size
        assert y.shape == (self.batch_size,)

    def test_cifar(self):
        ds = CIFAR10(base_settings.DATA_ROOT, download=True, train=False, transform=get_train_transform(True, 'cifar'))
        cifar_dl = self.dl_create_func(ds)
        x, y = next(iter(cifar_dl))
        assert x.shape == (self.batch_size, 3, 32, 32)


if __name__ == '__main__':
    unittest.main()
