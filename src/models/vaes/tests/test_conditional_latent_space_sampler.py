import unittest
import os.path as op

import numpy as np

from src.models.vaes import conditional_latent_space_sampler
from src.models.vaes.tests.test_latent_space_sampler import data_path


class ConditionalLatentSpaceSamplerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_trace_path = op.join(data_path(), 'latent_codes_clustering', 'sample_trace.pkl')
        cls.conditional_sampler = conditional_latent_space_sampler.ConditionalLatentSpaceSampler(cls.sample_trace_path)

        cls.sample_unobserved_tuples = [(0,), (1, 2, 5), (0, 9), (0, 3, 4, 6), (1, 2, 3, 4, 5)]

    def test_prepare_marginal_params_for_observed(self):
        cluster_index = 4
        data_dim = self.conditional_sampler.data_dim
        for unobs_tuple in self.sample_unobserved_tuples:
            marginal_params = self.conditional_sampler.prepare_marginal_params_for_observed(cluster_index, unobs_tuple)
            assert marginal_params['mean'].shape == (data_dim - len(unobs_tuple),)
            assert marginal_params['cov_chol'].shape == (data_dim - len(unobs_tuple), data_dim - len(unobs_tuple))
            assert 'df' in marginal_params

    def test_prepare_conditional_params_for_unobserved(self):
        cluster_index = 4
        joint_vec = np.squeeze(self.conditional_sampler.sample_latent_vecs_for_cluster(cluster_index, 1))

        for unobs_tuple in self.sample_unobserved_tuples:
            obs_vec = np.delete(joint_vec, unobs_tuple)
            cond_params = self.conditional_sampler.prepare_conditional_params_for_unobserved(cluster_index, unobs_tuple,
                                                                                             obs_vec)
            assert cond_params['mean'].shape == (len(unobs_tuple),)
            assert cond_params['cov_chol'].shape == (len(unobs_tuple), len(unobs_tuple))
            assert 'df' in cond_params
            assert np.all(np.isfinite(cond_params['mean']))
            assert np.all(np.isfinite(cond_params['cov_chol']))

    def test_sample_latent_vecs_with_unobserved_for_cluster(self):
        cluster_index, samples_num = 4, 20
        unobs_with_conditioning_flag = self.sample_unobserved_tuples + self.sample_unobserved_tuples
        unobs_with_conditioning_flag = zip(unobs_with_conditioning_flag,
                                           [True, False] * len(self.sample_unobserved_tuples))
        for unobs_tuple, use_conditioning in unobs_with_conditioning_flag:
            vecs = self.conditional_sampler.sample_latent_vecs_with_unobserved_for_cluster(cluster_index,
                                                                                           unobs_tuple,
                                                                                           samples_num,
                                                                                           use_conditioning=use_conditioning)

            assert vecs.shape == (samples_num, self.conditional_sampler.data_dim)
            assert np.all(np.isfinite(vecs))

            observed_indices = sorted(list(set(range(self.conditional_sampler.data_dim)) - set(unobs_tuple)))
            obs_part = vecs[:, observed_indices]

            assert all(np.array_equal(obs_part[i, :], obs_part[i + 1, :]) for i in range(samples_num - 1))

    def test_sample_factorized_latent_vecs_for_cluster(self):
        cluster_index, samples_num = 4, 20
        independent_vecs = self.conditional_sampler.sample_factorized_latent_vecs_for_cluster(cluster_index,
                                                                                              samples_num)
        assert np.all(np.isfinite(independent_vecs))
        assert independent_vecs.shape == (samples_num, self.conditional_sampler.data_dim)

    def test_sample_factorized_latent_vecs_from_mixture(self):
        samples_num = 20
        latent_vecs = self.conditional_sampler.sample_factorized_latent_vecs_from_mixture(samples_num)
        assert np.all(np.isfinite(latent_vecs))
        assert latent_vecs.shape == (samples_num, self.conditional_sampler.data_dim)


if __name__ == '__main__':
    unittest.main()
