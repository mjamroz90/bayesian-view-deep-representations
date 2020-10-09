import os.path as op
import random

from scipy import stats
import numpy as np
import tensorflow as tf
import choldate
from tensorflow_probability import distributions as tpd

from utils import prob_utils
from utils.logger import log
from utils import fs_utils
from src.models.clustering import cgs_utils


@log
class CollapsedGibbsSampler(object):

    def __init__(self, init_strategy, max_clusters_num, tf_shared=False, out_dir=None, **kwargs):
        self.init_strategy = init_strategy
        self.max_clusters_num = max_clusters_num

        self.a = kwargs['a'] if 'a' in kwargs else 1.0
        self.b = kwargs['b'] if 'b' in kwargs else 1.0

        self.alpha = self.__sample_alpha()
        self.logger.info("Sampled initial alpha, it's value: %.3f" % self.alpha)

        self.out_dir = out_dir
        self.tf_shared = tf_shared

        self.skip_epochs_logging = kwargs['skip_epochs_logging'] if 'skip_epochs_logging' in kwargs else 1

    def fit(self, iterations_num, data):
        n_points, curr_alpha = data.shape[0], self.alpha

        cluster_assignment, examples_assignment = self.get_initial_assignment(data)
        n_comps = len(cluster_assignment)

        mean_0, cov_0 = self.__initial_mean0_cov0(data, data, n_comps)
        nu_0 = cgs_utils.init_nu_0(data)

        cov_chol_0 = np.linalg.cholesky(cov_0)

        cluster_params = self.assign_initial_params(cluster_assignment, data, n_comps)

        self.logger.info("Initialized params for first assignment")
        self.logger.info("Chosen first assignment, clusters num: %d" % len(cluster_assignment))
        sess = tf.Session()

        if not self.tf_shared:
            tf_computation_manager = cgs_utils.TfCgsNonSharedComputationsManager(data.shape[1], mean_0,
                                                                                 cov_chol_0, nu_0, sess)
        else:
            init_values = {'means': np.array(cluster_params['mean']), 'cov_chols': np.array(cluster_params['cov_chol']),
                           'mean_0': mean_0, 'cov_chol_0': cov_chol_0}
            tf_computation_manager = cgs_utils.TfCgsSharedComputationsManager(data, len(cluster_assignment),
                                                                              sess, fs_utils.chol_date_lib_path(),
                                                                              init_values)
        mvn_log_pdf, mvn_phs = self.log_likelihood_tf(data)
        self.logger.info("Composed computational graph for tensorflow t-student pdf computations")

        ex_permutation = list(range(data.shape[0]))
        for iter_num in range(iterations_num):
            if self.tf_shared and iter_num > 0:
                params_for_ll = {'mean': res['mean'], 'cov_chol': res['cov_chol']}
            else:
                params_for_ll = cluster_params
            ll, ass_ll = self.data_log_likelihood(cluster_assignment, data, params_for_ll, nu_0, mvn_log_pdf, sess,
                                                  mvn_phs, curr_alpha)
            self.logger.info("Started %d epoch, current  clusters number: %d, current ll: %.2f, assignment ll: %.2f, "
                             "curr-alpha: %.2f " % (iter_num, len(cluster_assignment), ll, ass_ll, curr_alpha))

            for data_point_indx in ex_permutation:
                curr_alpha = self.__update_alpha(curr_alpha, n_points, len(cluster_assignment))
                data_point_cluster, cluster_removed = self.remove_assignment_for_data_point(data_point_indx,
                                                                                            data[data_point_indx, :],
                                                                                            cluster_assignment,
                                                                                            examples_assignment,
                                                                                            cluster_params)

                cluster_counts = [len(cluster_examples) for cluster_examples in cluster_assignment]
                if self.tf_shared:
                    # At last data point, fetch current means and covariances for clusters
                    return_curr_params = True if data_point_indx == ex_permutation[-1] else False
                    res = tf_computation_manager.sample_cluster_for_data_point(data_point_indx, data_point_cluster,
                                                                               cluster_removed, cluster_counts,
                                                                               curr_alpha,
                                                                               return_curr_params=return_curr_params)
                    z_indx = res['sampled_cluster']
                else:
                    z_indx = tf_computation_manager.sample_cluster_for_data_point(cluster_params, cluster_counts,
                                                                                  data[data_point_indx, :], n_points,
                                                                                  curr_alpha)
                # Very rare error that tf sampling returns index beyond the domain, because of numerical issues
                if z_indx > len(cluster_assignment):
                    z_indx = len(cluster_assignment)

                if not self.tf_shared:
                    self.update_sampled_cluster_params(z_indx, data[data_point_indx, :], cluster_params,
                                                       (mean_0, cov_chol_0), cluster_counts)

                if z_indx == len(cluster_assignment):
                    cluster_assignment.append({data_point_indx})
                else:
                    cluster_assignment[z_indx].add(data_point_indx)

                examples_assignment[data_point_indx] = z_indx
                if data_point_indx % 100 == 0:
                    self.logger.info("Iteration: %d, example: %d/%d, current clusters num: %d, alpha: %.2f" %
                                     (iter_num, data_point_indx, data.shape[0], len(cluster_assignment), curr_alpha))

            if self.out_dir is not None and iter_num % self.skip_epochs_logging == 0:
                self.save_model(params_for_ll, cluster_assignment, data, iter_num, curr_alpha, ll)
                self.logger.info("Saved model from iteration: %d" % iter_num)

            random.shuffle(ex_permutation)

        return params_for_ll

    def update_sampled_cluster_params(self, sampled_cluster, data_point, cluster_params, params_0,
                                      cluster_counts):
        if sampled_cluster == len(cluster_params['mean']):
            mean_0, cov_chol_0 = params_0
            new_mean, new_cov_chol = self.update_cluster_params(mean_0, np.copy(cov_chol_0), data_point, 0)
            cluster_params['mean'].append(new_mean)
            cluster_params['cov_chol'].append(new_cov_chol)
            self.logger.info("Sampled new cluster")
        else:
            current_mean = cluster_params['mean'][sampled_cluster]
            current_cov_chol = cluster_params['cov_chol'][sampled_cluster]

            new_mean, new_cov_chol = self.update_cluster_params(current_mean, current_cov_chol, data_point,
                                                                cluster_counts[sampled_cluster])
            cluster_params['mean'][sampled_cluster] = new_mean
            cluster_params['cov_chol'][sampled_cluster] = new_cov_chol

    def remove_assignment_for_data_point(self, data_point_indx, data_point, cluster_assignment, examples_assignment,
                                         cluster_params):
        data_point_cluster = examples_assignment[data_point_indx]
        cluster_assignment[data_point_cluster].remove(data_point_indx)

        # not necessary, but for the sake of clarification
        examples_assignment[data_point_indx] = -1

        if len(cluster_assignment[data_point_cluster]) == 0:
            self.logger.info("Removing cluster: %d" % data_point_cluster)
            del cluster_assignment[data_point_cluster]
            for cluster_idx, examples in enumerate(cluster_assignment):
                for ex in examples:
                    examples_assignment[ex] = cluster_idx

            if not self.tf_shared:
                del cluster_params['mean'][data_point_cluster]
                del cluster_params['cov_chol'][data_point_cluster]

            return data_point_cluster, True
        elif not self.tf_shared:
            cluster_mean = cluster_params['mean'][data_point_cluster]
            cluster_cov_chol = cluster_params['cov_chol'][data_point_cluster]
            cluster_n = len(cluster_assignment[data_point_cluster]) + 1

            new_cluster_mean, new_cluster_cov_chol = self.downdate_cluster_params(cluster_mean, cluster_cov_chol,
                                                                                  data_point, cluster_n)

            cluster_params['mean'][data_point_cluster] = new_cluster_mean
            cluster_params['cov_chol'][data_point_cluster] = new_cluster_cov_chol
            return data_point_cluster, False
        else:
            return data_point_cluster, False

    @staticmethod
    def log_likelihood_tf(data):
        data_ph = tf.placeholder(dtype=tf.float32, shape=data.shape)
        mean_ph = tf.placeholder(dtype=tf.float32, shape=(data.shape[1],))
        cov_chol_ph = tf.placeholder(dtype=tf.float32, shape=(data.shape[1], data.shape[1]))

        dist = tpd.MultivariateNormalTriL(loc=mean_ph, scale_tril=cov_chol_ph)
        return dist.log_prob(data_ph), {'mean': mean_ph, 'cov_chol': cov_chol_ph, 'data': data_ph}

    def data_log_likelihood(self, cluster_assignment, data, cluster_params, nu_0, mvn_log_pdf, sess,
                            tf_phs, curr_alpha):
        # sampling covariance from marginal p(sigma | D) for each cluster k=1,2,...,K
        # sampling mean from marginal p(mean | D) for each cluster k=1,2,...,K
        # sampling pi_k for each cluster k=1,2,...,K
        # p(sigma | D) = IW(S_n, nu_n)
        # p(mean | D) = t_student_pdf(mean | m_n, [1/(kappa_n)(nu_n-D+1)]S_n, nu_n-D+1)
        # p(pi | z) = Dir({alpha_k + \sum_{i=1}^N I(z_i=k) })
        clusters_num = len(cluster_assignment)
        examples_assignment = [0] * data.shape[0]
        for cluster, cluster_examples in enumerate(cluster_assignment):
            for ex in cluster_examples:
                examples_assignment[ex] = cluster

        data_dim = data.shape[1]
        means, cov_chols = self.__sample_marginals_for_mean_and_sigma(cluster_assignment, cluster_params, nu_0,
                                                                      data_dim)
        weights = self.__sample_weights(cluster_assignment, clusters_num, curr_alpha)
        data_log_pdfs = []
        for cov_chol, mean in zip(cov_chols, means):
            k_log_pdfs = sess.run(mvn_log_pdf, feed_dict={tf_phs['mean']: mean, tf_phs['cov_chol']: cov_chol,
                                                          tf_phs['data']: data})
            data_log_pdfs.append(k_log_pdfs)

        data_log_pdfs = np.array(data_log_pdfs, dtype=np.float32)
        # log ( weight[z_i] * P(x_i | z_i)) = log weight[z_i] + logP(x_i | z_i)
        examples_clusters_ll = np.expand_dims(np.log(weights), axis=-1) + data_log_pdfs
        ll = np.sum(examples_clusters_ll, axis=0).sum()

        assignment_ll = np.sum(examples_clusters_ll[examples_assignment, np.arange(data.shape[0])])
        return ll, assignment_ll

    @staticmethod
    def __sample_marginals_for_mean_and_sigma(cluster_assignment, cluster_params, nu_0, data_dim):
        sigmas_chols, means = [], []

        for k, k_examples in enumerate(cluster_assignment):
            k_mean, k_cov_chol = cluster_params['mean'][k], cluster_params['cov_chol'][k]
            nu_k, kappa_k = nu_0 + len(k_examples), cgs_utils.init_kappa_0() + len(k_examples)

            s_k = np.dot(k_cov_chol, k_cov_chol.T)
            sigma_k = stats.invwishart.rvs(df=nu_k, scale=s_k)
            mean_k = prob_utils.multivariate_t_rvs(k_mean, np.sqrt(1. / (kappa_k * (nu_k - data_dim + 1))) * k_cov_chol,
                                                   df=nu_k - data_dim + 1)

            sigmas_chols.append(np.linalg.cholesky(sigma_k))
            means.append(mean_k)

        return means, sigmas_chols

    @staticmethod
    def __sample_weights(cluster_assignment, k, alpha):
        alpha_k = np.ones(k) * (alpha / k)
        alpha_k += [len(c) for c in cluster_assignment]
        return stats.dirichlet.rvs(alpha=alpha_k, size=1)[0]

    @staticmethod
    def downdate_cluster_params(mean, cov_chol, data_point, n_cluster):
        kappa_0 = cgs_utils.init_kappa_0()
        new_mean = (mean * (kappa_0 + n_cluster) - data_point) / (kappa_0 + n_cluster - 1)

        u_vec = np.sqrt((kappa_0 + n_cluster) / (kappa_0 + n_cluster - 1)) * (data_point - mean).astype(
            np.float64)
        # overriding old covariance matrix
        current_cov_chol = cov_chol.astype(np.float64).T
        choldate.choldowndate(current_cov_chol, u_vec.copy())

        return new_mean.astype(np.float32), current_cov_chol.T.astype(np.float32)

    @staticmethod
    def update_cluster_params(mean, cov_chol, data_point, n_cluster):
        kappa_0 = cgs_utils.init_kappa_0()
        new_mean = (mean * (kappa_0 + n_cluster) + data_point) / (kappa_0 + n_cluster + 1)

        u_vec = np.sqrt((kappa_0 + n_cluster + 1) / (kappa_0 + n_cluster)) * (data_point - new_mean).astype(
            np.float64)
        current_cov_chol = cov_chol.astype(np.float64).T

        choldate.cholupdate(current_cov_chol, u_vec.copy())

        return new_mean.astype(np.float32), current_cov_chol.T.astype(np.float32)

    def get_initial_assignment(self, data):
        clusters_num = min(data.shape[0], self.max_clusters_num)
        init = np.random.randint(0, clusters_num, size=data.shape[0])
        clusters_examples = {}
        for example_idx, example_cluster in enumerate(init):
            if example_cluster in clusters_examples:
                clusters_examples[example_cluster].add(example_idx)
            else:
                clusters_examples[example_cluster] = {example_idx}

        cluster_assignment = [examples for _, examples in clusters_examples.items()]
        examples_assignment = [0] * data.shape[0]

        for cluster_idx, examples in enumerate(cluster_assignment):
            for ex in examples:
                examples_assignment[ex] = cluster_idx

        return cluster_assignment, examples_assignment

    def assign_initial_params(self, assignment, data, n_comps):
        cluster_params = {'mean': [], 'cov_chol': []}

        for cluster_num, examples in enumerate(assignment):
            cluster_data = data[list(examples), :]

            cluster_mean, cluster_cov_chol = self.initialize_params_for_samples(data, cluster_data, n_comps)
            cluster_params['mean'].append(cluster_mean)
            cluster_params['cov_chol'].append(cluster_cov_chol)

        return cluster_params

    def initialize_params_for_samples(self, data, cluster_data, n_comps):
        kappa_0 = cgs_utils.init_kappa_0()
        n_cluster = cluster_data.shape[0]
        sample_data_mean = np.mean(cluster_data, axis=0)

        cluster_mean_0, cluster_cov_0 = self.__initial_mean0_cov0(data, cluster_data, n_comps)

        cluster_mean = (kappa_0 * cluster_mean_0 + n_cluster * sample_data_mean) / (kappa_0 + n_cluster)

        cluster_cov = cluster_cov_0 + np.dot(cluster_data.T, cluster_data)
        cluster_cov += kappa_0 * np.outer(cluster_mean_0, cluster_mean_0)
        cluster_cov -= (kappa_0 + n_cluster) * np.outer(cluster_mean, cluster_mean)
        return cluster_mean, np.linalg.cholesky(cluster_cov)

    def __initial_mean0_cov0(self, data, cluster_data, n_comps):
        if self.init_strategy == 'init_per_init_cluster':
            cluster_mean_0 = self.init_mean(cluster_data)
            if cluster_data.shape[0] > 1:
                cluster_cov_0 = self.init_cov(cluster_data, n_comps)
            else:
                cluster_cov_0 = self.init_cov_eye(cluster_data)
        elif self.init_strategy == 'init_randomly':
            cluster_mean_0 = self.init_mean_random(cluster_data)
            cluster_cov_0 = self.init_cov_random(cluster_data)
        elif self.init_strategy == 'init_eye':
            cluster_mean_0 = self.init_mean(cluster_data)
            cluster_cov_0 = self.init_cov_eye(cluster_data)
        elif self.init_strategy == 'init_data_stats':
            cluster_mean_0, cluster_cov_0 = self.init_mean(data), self.init_cov(data, n_comps)
        else:
            raise ValueError("Unknown initialization: %s" % self.init_strategy)
        return cluster_mean_0, cluster_cov_0

    def __sample_alpha(self):
        return stats.gamma.rvs(self.a, self.b)

    def __update_alpha(self, old_alpha, n_points, k):
        u = stats.bernoulli.rvs(float(n_points) / (n_points + old_alpha))
        v = stats.beta.rvs(old_alpha + 1., n_points)

        new_alpha = np.random.gamma(self.a + k - 1 + u, 1. / (self.b - np.log(v)))
        return new_alpha

    @staticmethod
    def init_cov(data, num_components):
        data_mean = np.mean(data, axis=0)
        data_norm = data - np.expand_dims(data_mean, axis=0)

        data_var = np.dot(data_norm.T, data_norm) * (1. / data.shape[0])
        div_factor = np.power(num_components, 2. / data.shape[1])

        return np.diag(np.diag(data_var / div_factor))

    @staticmethod
    def init_cov_eye(data):
        return np.eye(data.shape[1], dtype=np.float64)

    @staticmethod
    def init_cov_random(data):
        d = data.shape[1]
        vec = np.random.randn(d, d)
        cov = np.dot(vec.T, vec)
        return cov

    @staticmethod
    def init_mean(data):
        return np.mean(data, axis=0)

    @staticmethod
    def init_mean_random(data):
        d = data.shape[1]
        return np.random.randn(d)

    def save_model(self, cluster_params, cluster_assignment, data, it_index, curr_alpha, ll):
        import pickle

        obj = {'cluster_assignment': cluster_assignment, 'cluster_params': cluster_params,
               'init_strategy': self.init_strategy, 'alpha': curr_alpha, 'll': ll}
        try:
            if it_index == 0:
                obj.update({'data': data})
            with open(op.join(self.out_dir, "cgs_%d.pkl" % it_index), 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            self.logger.error(e)
