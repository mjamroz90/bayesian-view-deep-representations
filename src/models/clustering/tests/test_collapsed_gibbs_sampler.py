import unittest

import numpy as np
import tensorflow as tf

from src.models.clustering import collapsed_gibbs_sampler
from src.models.clustering import cgs_utils
from utils.logger import log


@log
class CollapsedGibbsSamplerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data = np.random.randn(100, 5).astype(np.float64)
        cls.nu_0 = cls.sample_data.shape[1] + 2
        cls.gibbs_sampler = collapsed_gibbs_sampler.CollapsedGibbsSampler('mean0_cov0',
                                                                          max_clusters_num=cls.sample_data.shape[0])

        cls.kappa_0 = 0.01
        cls.mean_0, cls.cov_0 = cls.gibbs_sampler.init_mean(cls.sample_data), cls.gibbs_sampler.init_cov(
            cls.sample_data)

    def test_initial_assignment(self):
        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        assert len(cluster_assignment) <= self.sample_data.shape[0]
        assert len(cluster_assignment) > 0
        for i, ex_cluster in enumerate(examples_assignment):
            assert (i in cluster_assignment[ex_cluster])
        self.logger.info("cluster assignment: %s" % str(cluster_assignment))

    def test_assign_initial_params(self):
        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)
        assert (np.array(cluster_params['mean']).shape == (len(cluster_params['mean']), self.sample_data.shape[1]))
        assert (np.array(cluster_params['cov_chol']).shape == (len(cluster_params['cov_chol']),
                                                               self.sample_data.shape[1], self.sample_data.shape[1]))
        assert (len(cluster_params['mean']) == len(cluster_params['cov_chol']))

    def test_modify_cluster_params(self):
        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)

        some_cluster_index = [i for i, ass in enumerate(cluster_assignment) if len(ass) > 1][0]
        some_cluster_examples = list(cluster_assignment[some_cluster_index])

        some_cluster_mean = cluster_params['mean'][some_cluster_index]
        some_cluster_cov_chol = cluster_params['cov_chol'][some_cluster_index]
        index_to_remove, remaining_indices = some_cluster_examples[0], some_cluster_examples[1:]
        remaining_data = self.sample_data[remaining_indices, :]

        init_m, init_cov_chol = self.gibbs_sampler.initialize_params_for_samples(remaining_data, self.mean_0,
                                                                                 self.cov_0)
        down_m, down_cov_chol = self.gibbs_sampler.downdate_cluster_params(some_cluster_mean,
                                                                           some_cluster_cov_chol.copy(),
                                                                           self.sample_data[index_to_remove, :],
                                                                           len(some_cluster_examples))
        assert (np.sum(np.power(init_m - down_m, 2)) < 1e-14)
        assert (np.sum(np.power(init_cov_chol - down_cov_chol, 2)) < 1e-14)

        new_sample_index = list(cluster_assignment[0])[0]
        increased_data = self.sample_data[some_cluster_examples + [new_sample_index], :]
        init_m, init_cov_chol = self.gibbs_sampler.initialize_params_for_samples(increased_data, self.mean_0,
                                                                                 self.cov_0)
        upd_m, upd_cov_chol = self.gibbs_sampler.update_cluster_params(some_cluster_mean, some_cluster_cov_chol,
                                                                       self.sample_data[new_sample_index, :],
                                                                       len(some_cluster_examples))
        assert (np.sum(np.power(init_m - upd_m, 2)) < 1e-10)
        assert (np.sum(np.power(init_cov_chol - upd_cov_chol, 2)) < 1e-10)

    def test_remove_assignment_for_data_point(self):
        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)
        cluster_num = len(cluster_params['mean'])

        data_point_to_remove = list([ass for ass in cluster_assignment if len(ass) > 1][0])[0]
        old_data_point_cluster = examples_assignment[data_point_to_remove]
        self.gibbs_sampler.remove_assignment_for_data_point(data_point_to_remove,
                                                            self.sample_data[data_point_to_remove, :],
                                                            cluster_assignment, examples_assignment, cluster_params)

        assert (examples_assignment[data_point_to_remove] == -1)
        assert (data_point_to_remove not in cluster_assignment[old_data_point_cluster])
        assert (len(cluster_params['mean']) == cluster_num)
        assert (len(cluster_params['cov_chol']) == cluster_num)

        data_point_to_remove = list([ass for ass in cluster_assignment if len(ass) == 1][0])[0]
        self.gibbs_sampler.remove_assignment_for_data_point(data_point_to_remove,
                                                            self.sample_data[data_point_to_remove, :],
                                                            cluster_assignment, examples_assignment, cluster_params)
        assert (examples_assignment[data_point_to_remove] == -1)
        assert (len(cluster_params['mean']) == cluster_num - 1)
        assert (len(cluster_params['cov_chol']) == cluster_num - 1)

    def test_t_student_log_pdf_tf(self):
        sess = tf.Session()

        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)

        tf_phs = cgs_utils.create_input_phs(self.sample_data.shape[1])
        tf_phs['means'] = tf.placeholder(dtype=tf.float32, shape=(None, self.sample_data.shape[1]))
        tf_phs['cov_chols'] = tf.placeholder(dtype=tf.float32, shape=(None, self.sample_data.shape[1],
                                                                      self.sample_data.shape[1]))
        cluster_counts = [len(c) for c in cluster_assignment]
        nus = [self.nu_0 + cluster_n - self.sample_data.shape[1] + 1 for cluster_n in cluster_counts]

        t_student_pdfs = cgs_utils.t_student_log_pdf_tf(tf_phs['means'], tf_phs['cov_chols'],
                                                        tf_phs['data_point'],
                                                        tf_phs['nus'],
                                                        tf_phs['cluster_counts'])

        pdf_vals = sess.run(t_student_pdfs, feed_dict={tf_phs['data_point']: self.sample_data[1, :],
                                                       tf_phs['means']: cluster_params['mean'],
                                                       tf_phs['cov_chols']: cluster_params['cov_chol'],
                                                       tf_phs['nus']: nus,
                                                       tf_phs['cluster_counts']: cluster_counts})

        assert (pdf_vals.shape == (len(cluster_assignment),))
        assert (not np.all(np.isnan(pdf_vals)))

        big_data_chunk = self.sample_data[1:, :]
        big_cluster_mean, big_cluster_cov_chol = self.gibbs_sampler.initialize_params_for_samples(
            big_data_chunk, self.mean_0, self.cov_0)
        big_nu = self.nu_0 + big_data_chunk.shape[0] - big_data_chunk.shape[1] + 1

        big_cluster_pdf = sess.run(t_student_pdfs, feed_dict={tf_phs['data_point']: self.sample_data[0, :],
                                                              tf_phs['means']: [big_cluster_mean],
                                                              tf_phs['cov_chols']: [big_cluster_cov_chol],
                                                              tf_phs['nus']: [big_nu],
                                                              tf_phs['cluster_counts']: [big_data_chunk.shape[0]]})
        assert (big_cluster_pdf.shape == (1,))
        assert (not np.isnan(big_cluster_pdf[0]))

    def test_data_log_likelihood(self):
        sess = tf.Session()

        cluster_assignment, examples_assignment = self.gibbs_sampler.get_initial_assignment(self.sample_data)
        cluster_params = self.gibbs_sampler.assign_initial_params(cluster_assignment, self.sample_data)

        mvn_pdf, phs = self.gibbs_sampler.log_likelihood_tf(self.sample_data)

        ll, ass_ll = self.gibbs_sampler.data_log_likelihood(cluster_assignment, self.sample_data, cluster_params,
                                                            self.sample_data.shape[1] + 2, mvn_pdf, sess, phs,
                                                            self.gibbs_sampler.alpha)
        self.logger.info("Log likelihood: %.2f, assignment ll: %.2f" % (ll, ass_ll))
        assert (not np.isnan(ll))
        assert (not np.isnan(ass_ll))

    def test_fit_tf_non_shared(self):
        final_cluster_params = self.gibbs_sampler.fit(50, self.sample_data)
        assert len(final_cluster_params['mean']) > 0

    def test_fit_tf_shared(self):
        shared_gibbs_sampler = collapsed_gibbs_sampler.CollapsedGibbsSampler(init_strategy='mean0_cov0',
                                                                             max_clusters_num=self.sample_data.shape[0],
                                                                             tf_shared=True)
        final_cluster_params = shared_gibbs_sampler.fit(50, self.sample_data)
        assert len(final_cluster_params['mean']) > 0


if __name__ == '__main__':
    unittest.main()
