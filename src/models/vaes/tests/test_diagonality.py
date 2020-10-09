import os.path as op
import unittest

import tensorflow as tf
import numpy as np

from src.models.vaes import diagonality
from src.models.vaes.tests.test_latent_space_sampler import data_path
from utils import logger


@logger.log
class BhattacharyyaDistCalculatorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sample_trace_path = op.join(data_path(), 'latent_codes_clustering', 'sample_trace.pkl')
        cls.diagonality_checker = diagonality.BhattacharyyaDistCalculator(sample_trace_path)

    def test_compute_bhattacharyya_dist_for_clusters(self):
        clusters_dists = self.diagonality_checker.compute_bhattacharyya_dist_for_clusters()
        assert 'weighted_dist' in clusters_dists

        self.logger.info(clusters_dists)
        assert clusters_dists['weighted_dist'] > 0.


class SymmetricDKLCalculatorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_trace_path = op.join(data_path(), 'latent_codes_clustering', 'sample_trace.pkl')
        cls.dkl_calc = diagonality.DKLCalculator(10 ** 5, cls.sample_trace_path)

    def test_pq_ratio_for_samples(self):
        samples = tf.cast(self.dkl_calc.joint_distr.sample(sample_shape=(10 ** 5,)), dtype=tf.float32)
        ratios = self.dkl_calc.pq_ratio_for_samples(samples).numpy()

        assert ratios.shape == (10**5,)
        assert np.mean(ratios) > 0

    def test_calculate_joint_and_prod_dkl(self):
        dkl = self.dkl_calc.calculate_joint_and_prod_dkl()
        assert dkl > 0
        assert not np.isnan(dkl)

    def test_calculate_symmetric_dkl(self):
        dkl = self.dkl_calc.calculate_symmetric_dkl()
        assert dkl > 0
        assert not np.isnan(dkl)

    def test_dkl_correctness(self):
        t_student_params = self.dkl_calc.prepare_t_student_params(0)
        t_student_params['cov_chol'] = np.diag(np.diag(t_student_params['cov_chol']))
        joint_density_diag = self.dkl_calc.t_student_distr_from_params(t_student_params)
        marginal_densities = []

        for dim in range(self.dkl_calc.data_dim):
            dim_params = self.dkl_calc.prepare_marginal_params_for_dim(0, dim)
            marginal_densities.append(self.dkl_calc.t_student_distr_from_params(dim_params))

        joint_samples = tf.cast(joint_density_diag.sample(sample_shape=(10**5,)), dtype=tf.float32)
        joint_log_pdf = joint_density_diag.log_prob(joint_samples)
        prod_log_probs = []

        for i, i_marginal_density in enumerate(marginal_densities):
            i_log_probs = i_marginal_density.log_prob(tf.expand_dims(joint_samples[:, i], axis=-1))
            prod_log_probs.append(i_log_probs)

        prod_log_probs = tf.stack(prod_log_probs, axis=-1)
        prod_log_pdf = tf.reduce_sum(prod_log_probs, axis=-1)
        dkl = joint_log_pdf - prod_log_pdf

        dkl_mean = tf.reduce_mean(dkl).numpy()

        assert dkl_mean < 0.1


if __name__ == '__main__':
    unittest.main()
