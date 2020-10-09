import unittest
import os.path as op

import numpy as np

from src.models.vaes import latent_space_sampler


class LatentSpaceSamplerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_trace_path = op.join(data_path(), 'latent_codes_clustering', 'sample_trace.pkl')
        cls.latent_space_sampler = latent_space_sampler.LatentSpaceSampler(cls.sample_trace_path)

    def test_sample_latent_vecs_for_cluster(self):
        for cluster_index in range(self.latent_space_sampler.clusters_num):
            cluster_samples = self.latent_space_sampler.sample_latent_vecs_for_cluster(cluster_index, 20)
            assert cluster_samples.shape == (20, self.latent_space_sampler.data_dim)
            assert np.all(np.isfinite(cluster_samples))


def data_path():
    import src

    src_module_path = op.abspath(op.dirname(src.__file__))
    return op.join(src_module_path, '../data')


if __name__ == '__main__':
    unittest.main()
