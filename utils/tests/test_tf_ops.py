import unittest

import numpy as np
from utils import tf_ops


class TfOpsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shape = (100, 64)

    def test_sq_exp(self):
        a1 = np.random.randn(*self.shape)
        a2 = np.random.randn(*self.shape)

        kernel_mtx = tf_ops.sq_exp(a1, a2, 2. * self.shape[1] * 2.).numpy()
        assert kernel_mtx.sum() > 0.
        assert kernel_mtx.shape == (self.shape[0], self.shape[0])


if __name__ == '__main__':
    unittest.main()
