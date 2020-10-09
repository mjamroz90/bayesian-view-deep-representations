import unittest

import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow_probability import distributions as tpd

from src.models.clustering import cgs_utils
from src.models.clustering import collapsed_gibbs_sampler

from utils import fs_utils


class TfCgsSharedComputationsManagerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.sample_data = np.random.randn(100, 5).astype(np.float64)
        cls.alpha = stats.gamma.rvs(1, 1)
        cls.sess = tf.Session()
        cls.lib_path = fs_utils.chol_date_lib_path()
        cls.gibbs_sampler = collapsed_gibbs_sampler.CollapsedGibbsSampler('init_randomly',
                                                                          cls.sample_data.shape[0], None)

    def test_t_student_log_pdf(self):
        mean_ph = tf.placeholder(dtype=tf.float32, shape=(1, 5))
        cov_chol_ph = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5))
        input_phs = cgs_utils.create_input_phs(5)
        nu_0, kappa_0 = cgs_utils.init_nu_0(self.sample_data), cgs_utils.init_kappa_0()

        mean_arr = self.gibbs_sampler.init_mean_random(self.sample_data)
        cov_chol_arr = np.linalg.cholesky(self.gibbs_sampler.init_cov_random(self.sample_data))
        cluster_n = 1.
        nu_val = 3. + cluster_n
        kappa = kappa_0 + cluster_n

        data_point = tf.constant(self.sample_data[0, :], dtype=tf.float32)

        scaled_cov_chol_ph = np.sqrt((kappa + 1.) / (kappa * nu_val)) * cov_chol_ph

        log_pdf = cgs_utils.t_student_log_pdf_tf(mean_ph, cov_chol_ph, data_point, input_phs['nus'],
                                                 input_phs['cluster_counts'])
        t_student_distr = tpd.MultivariateStudentTLinearOperator(df=input_phs['nus'][0], loc=mean_ph[0],
                                                                 scale=tf.linalg.LinearOperatorLowerTriangular(
                                                                     scaled_cov_chol_ph[0]))

        log_pdf_val = self.sess.run(log_pdf[0], feed_dict={mean_ph: [mean_arr], cov_chol_ph: [cov_chol_arr],
                                                           input_phs['nus']: [nu_val],
                                                           input_phs['cluster_counts']: [cluster_n]})
        gt_log_pdf_val = self.sess.run(t_student_distr.log_prob(data_point),
                                       feed_dict={mean_ph: [mean_arr], cov_chol_ph: [cov_chol_arr],
                                                  input_phs['nus']: [nu_val]})

        assert np.power(log_pdf_val - gt_log_pdf_val, 2) < 1e-12

    def test_sample_cluster_for_data_point(self):
        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)

        mean0 = self.gibbs_sampler.init_mean_random(self.sample_data)
        cov_chol0 = np.linalg.cholesky(self.gibbs_sampler.init_cov_random(self.sample_data))

        init_values = {'means': np.array(cluster_params['mean']), 'cov_chols': np.array(cluster_params['cov_chol']),
                       'mean_0': mean0, 'cov_chol_0': cov_chol0}
        tf_shared_manager = cgs_utils.TfCgsSharedComputationsManager(self.sample_data, len(cluster_assignment),
                                                                     self.sess, self.lib_path, init_values)

        data_point_cluster, cluster_removed = self.gibbs_sampler.remove_assignment_for_data_point(
            0, self.sample_data[0, :], cluster_assignment, examples_assignment, cluster_params)
        cluster_counts = [len(c) for c in cluster_assignment]

        res = tf_shared_manager.sample_cluster_for_data_point(0, data_point_cluster, cluster_removed,
                                                              cluster_counts, self.alpha, True)

        sampled_cluster, new_m, new_c = res['sampled_cluster'], res['mean'], res['cov_chol']
        assert 0 < sampled_cluster <= len(cluster_assignment)
        if cluster_removed:
            assert tf_shared_manager.active_clusters[data_point_cluster] is False

        self.__check_updated_params(self.sample_data[0, :], cluster_params, sampled_cluster, cluster_counts,
                                    (mean0, cov_chol0), (new_m, new_c))

    def __check_updated_params(self, data_point, cluster_params, sampled_cluster, cluster_counts,
                               params0, new_params):
        self.gibbs_sampler.update_sampled_cluster_params(sampled_cluster, data_point, cluster_params,
                                                         params0, cluster_counts)

        sampled_cluster_mean = cluster_params['mean'][sampled_cluster]
        sampled_cluster_cov_chol = cluster_params['cov_chol'][sampled_cluster]

        tf_cluster_mean, tf_cluster_cov_chol = new_params[0][sampled_cluster], new_params[1][sampled_cluster]

        assert np.sum(np.power(tf_cluster_mean - sampled_cluster_mean, 2)) < 1e-10
        assert np.sum(np.power(tf_cluster_cov_chol - sampled_cluster_cov_chol, 2)) < 1e-10


if __name__ == '__main__':
    unittest.main()
