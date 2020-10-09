import unittest
import os.path as op

import numpy as np

from src.models.clustering import entropy_estimation


class EntropyEstimatorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_trace_path = op.join(data_path(), 'latent_codes_clustering', 'sample_trace.pkl')
        cls.data_trace_path = op.join(data_path(), 'latent_codes_clustering', 'cgs_0.pkl')
        cls.samples_num = 10000
        cls.diff_estimator = entropy_estimation.EntropyEstimator(cls.sample_trace_path, cls.samples_num, 'differential')
        cls.rel_estimator = entropy_estimation.EntropyEstimator(cls.sample_trace_path, cls.samples_num, 'relative',
                                                                **{'data_trace_path': cls.data_trace_path})

    def test_sample_latent_z(self):
        latent_z = self.diff_estimator.sample_latent_z(self.samples_num)

        assert latent_z.shape[0] == self.samples_num

    def test_sample_points_from_mixture(self):
        sampled_points = self.diff_estimator.sample_points_from_mixture()

        assert sampled_points.shape == (self.samples_num, self.diff_estimator.data_dim)

    def test_estimate_diff_entropy_with_sampling(self):
        entropy_val = self.diff_estimator.estimate_entropy_with_sampling()

        assert np.isfinite(entropy_val)

    def test_estimate_relative_entropy_with_sampling(self):
        entropy_val = self.rel_estimator.estimate_entropy_with_sampling()

        assert np.isfinite(entropy_val)


def data_path():
    import src

    src_module_path = op.abspath(op.dirname(src.__file__))
    return op.join(src_module_path, '../data')


if __name__ == '__main__':
    unittest.main()
