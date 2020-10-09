from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python import math
import numpy as np
from scipy import special

from utils.logger import log


def t_student_log_pdf_tf(mean_matrix, chol_cov_matrices, data_point, nus, cluster_counts):
    data_dim = tf.constant(data_point.get_shape()[0].value, dtype=tf.float32)
    kappas = tf.constant(init_kappa_0()) + cluster_counts

    scale_fact = tf.expand_dims(tf.expand_dims((kappas + 1.) / (kappas * nus), axis=-1), axis=-1)
    chol_cov_scaled = tf.sqrt(scale_fact) * chol_cov_matrices

    chol_cov_diagonals = tf.matrix_diag_part(chol_cov_scaled)
    log_dets_sqrt = tf.reduce_sum(tf.log(chol_cov_diagonals), axis=-1)

    data_point_norm = tf.expand_dims(data_point, axis=0) - mean_matrix
    vecs = tf.squeeze(tf.linalg.triangular_solve(chol_cov_scaled, tf.expand_dims(data_point_norm, axis=-1), lower=True))
    vecs_norm = tf.norm(vecs, axis=-1)

    num = tf.math.lgamma((nus + data_dim) / 2.)

    denom = tf.math.lgamma(nus / 2.) + (data_dim / 2.) * (tf.log(nus) + np.log(np.pi))
    denom += log_dets_sqrt
    denom += ((nus + data_dim) / 2.) * math.log1psquare(vecs_norm / tf.sqrt(nus))

    return num - denom


def create_input_phs(data_dim):
    data_point_ph = tf.placeholder(dtype=tf.float32, shape=(data_dim,))
    nus = tf.placeholder(dtype=tf.float32, shape=(None,))
    cluster_counts = tf.placeholder(dtype=tf.float32, shape=(None,))
    return {'data_point': data_point_ph, 'nus': nus, 'cluster_counts': cluster_counts}


def init_kappa_0():
    return 0.01


def init_nu_0(data):
    return data.shape[1] + 2


def init_cov(data):
    data_mean = np.mean(data, axis=0)
    data_norm = data - np.expand_dims(data_mean, axis=0)

    data_var = np.dot(data_norm.T, data_norm) * (1. / data.shape[0])

    return np.diag(np.diag(data_var))


class TfCgsNonSharedComputationsManager(object):

    def __init__(self, data_dim, mean_0, cov_chol_0, nu_0, sess):
        self.data_dim = data_dim
        self.nu_0 = nu_0
        self.mean_0 = mean_0
        self.cov_chol_0 = cov_chol_0

        self.sess = sess

        self.input_phs = create_input_phs(self.data_dim)
        self.input_phs['cov_chols'] = tf.placeholder(dtype=tf.float32, shape=(None, data_dim, data_dim))
        self.input_phs['means'] = tf.placeholder(dtype=tf.float32, shape=(None, data_dim))

        self.t_student_log_pdfs_for_clusters = t_student_log_pdf_tf(self.input_phs['means'],
                                                                    self.input_phs['cov_chols'],
                                                                    self.input_phs['data_point'],
                                                                    self.input_phs['nus'],
                                                                    self.input_phs['cluster_counts'])

        self.cluster_log_probs = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.sample_cluster_op = tf.random.categorical(logits=[self.cluster_log_probs], num_samples=1)[0][0]

    def sample_cluster_for_data_point(self, cluster_params, cluster_counts, data_point, n_points, alpha):
        cluster_probs = np.array([float(c) / (n_points + alpha - 1) for c in cluster_counts],
                                 dtype=np.float32)
        # p(x_i | x_{-i}{k}) - posterior predictive
        nus = [self.nu_0 + cluster_n - self.data_dim + 1 for cluster_n in cluster_counts]
        post_log_preds = self.sess.run(self.t_student_log_pdfs_for_clusters,
                                       feed_dict={self.input_phs['means']: cluster_params['mean'],
                                                  self.input_phs['cov_chols']: cluster_params['cov_chol'],
                                                  self.input_phs['data_point']: data_point,
                                                  self.input_phs['nus']: nus,
                                                  self.input_phs['cluster_counts']: cluster_counts})
        # p(z_i | z_{-i}, x, params) \prop to p(z_i | z_{-i}) * p(x_i | x_{-i}{k})
        cluster_log_preds_unnorm = np.log(cluster_probs) + post_log_preds

        new_cluster_prob = alpha / (n_points + alpha - 1)
        post_log_pred_new_cluster = self.sess.run(self.t_student_log_pdfs_for_clusters,
                                                  feed_dict={
                                                      self.input_phs['means']: [self.mean_0],
                                                      self.input_phs['cov_chols']: [self.cov_chol_0],
                                                      self.input_phs['data_point']: data_point,
                                                      self.input_phs['nus']: [self.nu_0 - self.data_dim + 1],
                                                      self.input_phs['cluster_counts']: [0]})
        new_cluster_log_prob_unnorm = np.log(new_cluster_prob) + post_log_pred_new_cluster[0]

        all_probs_log_unnormalized = np.append(cluster_log_preds_unnorm, [new_cluster_log_prob_unnorm])
        norm_const = special.logsumexp(all_probs_log_unnormalized)
        all_probs_log_normalized = all_probs_log_unnormalized - norm_const

        res = self.sess.run(self.sample_cluster_op, feed_dict={self.cluster_log_probs: all_probs_log_normalized})

        return res


@log
class TfCgsSharedComputationsManager(object):

    def __init__(self, data, init_clusters_num, sess, chol_date_lib_path, init_values):
        self.data = data
        self.data_dim = data.shape[1]
        self.init_clusters_num = init_clusters_num
        self.new_clusters_added = 0
        self.nu0 = tf.constant(init_nu_0(self.data), dtype=tf.float32)
        self.kappa0 = tf.constant(init_kappa_0(), dtype=tf.float32)

        self.sess = sess
        self.chol_up, self.chol_down = self.__load_update_ops(chol_date_lib_path)

        with tf.variable_scope('cgs_params'):
            self.data_sv = tf.get_variable(name='data', shape=self.data.shape, dtype=tf.float32)
            self.cov_chols = tf.get_variable(name='cov_chols', dtype=tf.float32, validate_shape=False,
                                             initializer=init_values['cov_chols'].astype(np.float32))

            self.means = tf.get_variable(name='means', dtype=tf.float32, validate_shape=False,
                                         initializer=init_values['means'].astype(np.float32))

            self.mean_0 = tf.get_variable(name='mean_0', shape=(self.data_dim,), dtype=tf.float32)
            self.cov_chol_0 = tf.get_variable(name='cov_chol_0', shape=(self.data_dim, self.data_dim),
                                              dtype=tf.float32)

        self.active_clusters = [True] * self.init_clusters_num
        self.new_old_cluster_mapping = list(range(init_clusters_num))

        self.sess.run([self.cov_chols.initializer, self.means.initializer, self.mean_0.assign(init_values['mean_0']).op,
                       self.cov_chol_0.assign(init_values['cov_chol_0']).op, self.data_sv.assign(self.data).op])

        self.logger.info("Assigned initial values to covariances, means and data points")
        self.sample_cluster_phs = self.tf_sample_cluster_for_data_point_graph()

    def sample_cluster_for_data_point(self, data_point_indx, curr_data_point_cluster, cluster_is_removed,
                                      cluster_counts, alpha, return_curr_params=False):

        old_data_point_cluster = self.new_old_cluster_mapping[curr_data_point_cluster]
        # cluster is already removed, data_point_indx was the only example, so cluster mask needs to be updated
        if cluster_is_removed:
            self.active_clusters[old_data_point_cluster] = False
            del self.new_old_cluster_mapping[curr_data_point_cluster]
            cluster_n = 0
            self.logger.info("Updated info about cluster to be removed: %d, clusters remained: %d" %
                             (curr_data_point_cluster, len(self.new_old_cluster_mapping)))
        else:
            cluster_n = cluster_counts[curr_data_point_cluster] + 1

        cluster_counts = [float(c) for c in cluster_counts]
        update_params_op = self.sample_cluster_phs['params_update_ops'] if return_curr_params \
            else [o.op for o in self.sample_cluster_phs['params_update_ops']]
        sampled_cluster, new_m, new_c = self.sess.run([self.sample_cluster_phs['sampled_cluster']] + update_params_op,
                                                      feed_dict={
                                                          self.sample_cluster_phs['data_point_index']: data_point_indx,
                                                          self.sample_cluster_phs['old_data_point_cluster']:
                                                              old_data_point_cluster,
                                                          self.sample_cluster_phs['cluster_counts']: cluster_counts,
                                                          self.sample_cluster_phs['cluster_mask']: self.active_clusters,
                                                          self.sample_cluster_phs['cluster_n']: float(cluster_n),
                                                          self.sample_cluster_phs['alpha']: alpha,
                                                          self.sample_cluster_phs['new_old_cluster_mapping']:
                                                              self.new_old_cluster_mapping})

        # Update mapping, if new cluster will be added
        if sampled_cluster == len(cluster_counts):
            self.new_clusters_added += 1
            self.new_old_cluster_mapping.append(self.init_clusters_num + self.new_clusters_added - 1)
            self.active_clusters.append(True)
            self.logger.info("Added new cluster: %d" % sampled_cluster)

        res = {'sampled_cluster': sampled_cluster}
        if return_curr_params:
            res.update({'mean': new_m[self.active_clusters, :], 'cov_chol': new_c[self.active_clusters, :, :]})

        return res

    def tf_sample_cluster_for_data_point_graph(self):
        n_points = tf.constant(self.data.shape[0], dtype=tf.float32)
        data_point_indx_ph = tf.placeholder(dtype=tf.int32, shape=())
        alpha_ph = tf.placeholder(dtype=tf.float32, shape=())
        old_data_point_cluster_ph = tf.placeholder(dtype=tf.int32, shape=())
        cluster_counts_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        cluster_mask = tf.placeholder(dtype=tf.bool, shape=(None,))
        curr_cluster_n = tf.placeholder(dtype=tf.float32, shape=())
        new_old_clusters_mapping = tf.placeholder(dtype=tf.int32, shape=(None,))

        data_point = self.data_sv[data_point_indx_ph, :]

        # conditional Downdating old cluster params
        down_means, down_cov_chols = tf.cond(tf.equal(curr_cluster_n, 0.), lambda: (self.means, self.cov_chols),
                                             lambda: self.downdate_cluster_params(old_data_point_cluster_ph, data_point,
                                                                                  curr_cluster_n))

        self.means, self.cov_chols = tf.assign(self.means, down_means), tf.assign(self.cov_chols, down_cov_chols)
        nus = self.nu0 - tf.constant(self.data_dim - 1, dtype=tf.float32) + cluster_counts_ph

        active_means = tf.boolean_mask(self.means, cluster_mask)
        active_cov_chols = tf.boolean_mask(self.cov_chols, cluster_mask)

        cluster_probs = cluster_counts_ph / (n_points + alpha_ph - 1.)
        pred_post_log_probs = t_student_log_pdf_tf(active_means, active_cov_chols, data_point,
                                                   nus, cluster_counts_ph)

        cluster_log_preds_unnorm = tf.log(cluster_probs) + pred_post_log_probs

        new_cluster_prob = alpha_ph / (n_points + alpha_ph - 1.)
        post_log_pred_new_cluster = t_student_log_pdf_tf(tf.expand_dims(self.mean_0, axis=0),
                                                         tf.expand_dims(self.cov_chol_0, axis=0),
                                                         data_point, 3., 0.)

        new_cluster_log_prob_unnorm = tf.log(new_cluster_prob) + post_log_pred_new_cluster[0]
        all_logs_probs_unnormalized = tf.concat([cluster_log_preds_unnorm, [new_cluster_log_prob_unnorm]], axis=0)

        norm_const = tf.reduce_logsumexp(all_logs_probs_unnormalized)
        all_probs_log_normalized = all_logs_probs_unnormalized - norm_const

        sampled_cluster = tf.cast(tf.squeeze(tf.random.categorical(logits=[all_probs_log_normalized], num_samples=1)),
                                  tf.int32)
        upd_means, upd_cov_chols = self.update_cluster_params(sampled_cluster, data_point,
                                                              cluster_counts_ph, new_old_clusters_mapping)
        new_means_op = tf.assign(self.means, upd_means, validate_shape=False)
        new_cov_chols_op = tf.assign(self.cov_chols, upd_cov_chols, validate_shape=False)

        return {'sampled_cluster': sampled_cluster, 'params_update_ops': [new_means_op, new_cov_chols_op],
                'data_point_index': data_point_indx_ph, 'alpha': alpha_ph,
                'old_data_point_cluster': old_data_point_cluster_ph,
                'cluster_counts': cluster_counts_ph,
                'cluster_mask': cluster_mask, 'cluster_n': curr_cluster_n,
                'new_old_cluster_mapping': new_old_clusters_mapping}

    def downdate_cluster_params(self, old_data_point_cluster, data_point, cluster_n):
        mean, cov_chol = self.means[old_data_point_cluster, :], self.cov_chols[old_data_point_cluster, :, :]
        new_mean = (mean * (self.kappa0 + cluster_n) - data_point) / (self.kappa0 + cluster_n - 1.)

        u_vec = tf.sqrt((self.kappa0 + cluster_n) / (self.kappa0 + cluster_n - 1.)) * (data_point - mean)
        new_cov_chol = self.chol_down(cov_chol, u_vec)

        new_means = tf.scatter_update(self.means, [old_data_point_cluster], [new_mean])
        new_cov_chols = tf.scatter_update(self.cov_chols, [old_data_point_cluster], [new_cov_chol])

        return new_means, new_cov_chols

    def update_cluster_params(self, new_data_point_cluster, data_point, cluster_counts,
                              new_old_cluster_mapping):
        def handle_new_cluster():
            new_mean_ = (self.mean_0 * self.kappa0 + data_point) / (self.kappa0 + 1)
            u_vec = tf.sqrt((self.kappa0 + 1) / self.kappa0) * (data_point - new_mean_)
            new_cov_chol_ = self.chol_up(self.cov_chol_0, u_vec)

            new_means = tf.concat([self.means, [new_mean_]], axis=0)
            new_cov_chols = tf.concat([self.cov_chols, [new_cov_chol_]], axis=0)

            return new_means, new_cov_chols

        def handle_existing_cluster():
            abs_cluster_indx = new_old_cluster_mapping[new_data_point_cluster]
            mean, cov_chol = self.means[abs_cluster_indx, :], self.cov_chols[abs_cluster_indx, :, :]
            cluster_n = cluster_counts[new_data_point_cluster]

            new_mean_ = (mean * (self.kappa0 + cluster_n) + data_point) / (self.kappa0 + cluster_n + 1)

            u_vec = tf.sqrt((self.kappa0 + cluster_n + 1) / (self.kappa0 + cluster_n)) * (data_point - new_mean_)
            new_cov_chol_ = self.chol_up(cov_chol, u_vec)

            new_means = tf.scatter_update(self.means, [abs_cluster_indx], [new_mean_])
            new_cov_chols = tf.scatter_update(self.cov_chols, [abs_cluster_indx], [new_cov_chol_])

            return new_means, new_cov_chols

        # with tf.control_dependencies(
        #        [tf.print("Sampled cluster", new_data_point_cluster, tf.shape(cluster_counts)[0])]):
        return tf.cond(tf.equal(tf.shape(cluster_counts)[0], new_data_point_cluster),
                       handle_new_cluster, handle_existing_cluster)

    @staticmethod
    def __load_update_ops(path):
        lib = tf.load_op_library(path)
        chol_up, chol_down = lib.chol_update, lib.chol_downdate
        return chol_up, chol_down
