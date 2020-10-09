import unittest
import torch

import numpy as np

from src.models import cca_cnn


class CcaCnnsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cca_imnet_model = cca_cnn.CCACNN('imagenet')
        cls.cca128_imnet_model = cca_cnn.CCACNN8x128('imagenet')

        cls.cca_cifar_model = cca_cnn.CCACNN('cifar')
        cls.cca128_cifar_model = cca_cnn.CCACNN8x128('cifar')

    def test_imagenet_forward(self):
        sample_input = torch.from_numpy(np.random.randn(16, 3, 42, 42).astype(np.float32))
        output = self.cca_imnet_model(sample_input)
        output128 = self.cca128_imnet_model(sample_input)
        assert list(output.shape)[-1] == 100 and list(output128.shape)[-1] == 100

    def test_cifar_forward(self):
        sample_input = torch.from_numpy(np.random.randn(16, 3, 32, 32).astype(np.float32))
        output = self.cca_cifar_model(sample_input)
        output128 = self.cca128_cifar_model(sample_input)
        assert list(output.shape)[-1] == 10 and list(output128.shape)[-1] == 10


if __name__ == '__main__':
    unittest.main()
