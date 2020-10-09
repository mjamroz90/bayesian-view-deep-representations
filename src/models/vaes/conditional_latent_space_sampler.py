import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tpd
from scipy import linalg
from src.models.vaes import latent_space_sampler


class ConditionalLatentSpaceSampler(latent_space_sampler.LatentSpaceSampler):

    def __run_sampling(self, t_student_params, samples_num):
        return self.t_student_sample(t_student_params['df'], t_student_params['mean'],
                                     t_student_params['cov_chol'], samples_num).numpy()

    @tf.function
    def t_student_sample(self, dof_t, mean_t, cov_chol_t, samples_num_t):
        t_student_distr = tpd.MultivariateStudentTLinearOperator(df=dof_t, loc=mean_t,
                                                                 scale=tf.linalg.LinearOperatorLowerTriangular(
                                                                     cov_chol_t))
        t_distr_samples = t_student_distr.sample(sample_shape=samples_num_t)
        return t_distr_samples

    def sample_latent_vecs_with_unobserved_for_cluster(self, cluster_index, unobserved_indices, samples_num,
                                                       use_conditioning=True):
        assert all(0 <= i < self.data_dim for i in unobserved_indices)

        unobserved_indices = sorted(list(unobserved_indices))
        observed_indices = sorted(list(set(range(self.data_dim)) - set(unobserved_indices)))

        marginal_t_student_params = self.prepare_marginal_params_for_observed(cluster_index, unobserved_indices)
        obs_vec_sampled = np.squeeze(self.__run_sampling(marginal_t_student_params, 1))
        self.logger.info("Sampled observed part of the final vector of shape: %s" % str(obs_vec_sampled.shape))

        if use_conditioning:
            unobs_t_student_params = self.prepare_conditional_params_for_unobserved(cluster_index, unobserved_indices,
                                                                                    obs_vec_sampled)
        else:
            # unobserved dimensions are being sampled from marginal distribution - not conditional
            unobs_t_student_params = self.prepare_marginal_params_for_observed(cluster_index, observed_indices)

        unobs_vec_sampled = self.__run_sampling(unobs_t_student_params, samples_num)
        self.logger.info("Sampled unobserved part of the final output, shape: %s" % str(unobs_vec_sampled.shape))

        final_vec = np.zeros((samples_num, self.data_dim), dtype=np.float32)
        final_vec[:, observed_indices] = obs_vec_sampled
        final_vec[:, unobserved_indices] = unobs_vec_sampled

        return final_vec

    def sample_factorized_latent_vecs_from_mixture(self, samples_num):
        dims_mixture_distributions = self.prepare_t_student_marginals_prod()

        dims_samples = [m.sample(sample_shape=(samples_num,)) for m in dims_mixture_distributions]
        dims_concatenated = tf.concat(dims_samples, axis=-1)
        self.logger.info("Sampled individual latent dimensions from mixture distributions of shape: %s" %
                         str(dims_concatenated.get_shape().as_list()))

        return dims_concatenated.numpy()

    def sample_factorized_latent_vecs_for_cluster(self, cluster_index, samples_num):
        assert cluster_index < self.clusters_num

        t_student_params = self.prepare_t_student_params(cluster_index)
        diag_cov_chol = np.diag(np.diag(t_student_params['cov_chol']))

        t_student_params.update({'cov_chol': diag_cov_chol})
        return self.__run_sampling(t_student_params, samples_num)

    def prepare_conditional_params_for_unobserved(self, cluster_index, unobserved_indices, observed_vec):
        assert observed_vec.ndim == 1

        unobserved_indices = list(unobserved_indices)
        observed_indices = sorted(list(set(range(self.data_dim)) - set(unobserved_indices)))
        obs_dim = observed_vec.shape[0]

        assert obs_dim + len(unobserved_indices) == self.data_dim
        joint_params = self.prepare_t_student_params(cluster_index)
        joint_mean, joint_cov_chol = joint_params['mean'], joint_params['cov_chol']
        joint_cov = np.dot(joint_cov_chol, joint_cov_chol.T)
        marginal_params = self.prepare_marginal_params_for_observed(cluster_index, unobserved_indices)

        # Using results from paper: On the Conditional Distribution of the Multivariate t Distribution, P. Ding
        # p(X_2 | X_1) = t-student_p2 (mu_2|1, \frac{nu + d1}{nu + p_1} \Sigma_{22|1}, nu + p_1), where
        # d1 = (X_1 - mu_1).T \Sigma_{11}^-1 (X_1 - mu_1) - Mahalanobis distance between X_1 and \mu_1
        # mu_2|1 = mu_2 + \Sigma_{21} \Sigma_{11}^-1 (X_1 - mu_1)
        # \Sigma_{22|1} = \Sigma_{22} - \Sigma_{21} \Sigma_{11}^-1 \Sigma_{12}

        # A^-1 = (L^-1).T (L^-1)
        obs_vec_norm = observed_vec - marginal_params['mean']
        sigma_11_inv_chol = linalg.solve_triangular(marginal_params['cov_chol'], np.eye(obs_dim, dtype=np.float32),
                                                    lower=True)

        sigma_11_inv = np.dot(sigma_11_inv_chol.T, sigma_11_inv_chol)

        mu_2 = joint_mean[unobserved_indices]
        sigma_21 = (joint_cov[unobserved_indices, :])[:, observed_indices]
        sigma_22 = (joint_cov[unobserved_indices, :])[:, unobserved_indices]
        sigma_12 = (joint_cov[observed_indices, :])[:, unobserved_indices]
        mu_21 = mu_2 + np.dot(np.dot(sigma_21, sigma_11_inv), obs_vec_norm)

        d1 = float(np.dot(np.dot(obs_vec_norm, sigma_11_inv), obs_vec_norm))
        sigma_221 = sigma_22 - np.dot(np.dot(sigma_21, sigma_11_inv), sigma_12)

        cond_mean = mu_21
        cond_cov = (float(marginal_params['df'] + d1) / (marginal_params['df'] + obs_dim)) * sigma_221
        cond_df = marginal_params['df'] + obs_dim

        return {'mean': cond_mean, 'df': cond_df, 'cov_chol': np.linalg.cholesky(cond_cov)}

    def prepare_marginal_params_for_observed(self, cluster_index, unobserved_indices):
        # Assumption that unobserved_indices are sorted here
        t_student_joint_params = self.prepare_t_student_params(cluster_index)
        marginal_mean = np.delete(t_student_joint_params['mean'], unobserved_indices)

        # Need to have a full covariance matrix in order to remove unobserved rows and columns
        joint_cov = np.dot(t_student_joint_params['cov_chol'], t_student_joint_params['cov_chol'].T)
        marginal_cov_axis0 = np.delete(joint_cov, unobserved_indices, axis=0)
        marginal_cov = np.delete(marginal_cov_axis0, unobserved_indices, axis=1)

        marginal_cov_chol = np.linalg.cholesky(marginal_cov)

        return {'df': t_student_joint_params['df'], 'mean': marginal_mean, 'cov_chol': marginal_cov_chol}

    def prepare_marginal_params_for_dim(self, cluster_index, dim_index):
        t_student_joint_params = self.prepare_t_student_params(cluster_index)
        means = t_student_joint_params['mean']
        cov_chols = t_student_joint_params['cov_chol']
        return {'df': t_student_joint_params['df'], 'mean': [means[dim_index]],
                'cov_chol': [[cov_chols[dim_index, dim_index]]]}

    def prepare_t_student_marginals_prod(self):
        cluster_weights_unnorm = self.get_cluster_unnormalized_weights()
        cluster_weights_norm = cluster_weights_unnorm / np.sum(cluster_weights_unnorm)

        cat_distr = tpd.Categorical(probs=cluster_weights_norm.astype(np.float32))
        dims_mixture_distributions = []

        for dim_index in range(self.data_dim):
            dim_mixture_cluster_params = [self.prepare_marginal_params_for_dim(cluster_index, dim_index)
                                          for cluster_index in range(self.clusters_num)]
            dim_dofs = tf.constant([d['df'] for d in dim_mixture_cluster_params], dtype=tf.float32)
            dim_means = tf.stack([d['mean'] for d in dim_mixture_cluster_params])
            dim_cov_chols = tf.stack([d['cov_chol'] for d in dim_mixture_cluster_params])

            dim_t_student_distr = tpd.MultivariateStudentTLinearOperator(df=dim_dofs, loc=dim_means,
                                                                         scale=tf.linalg.LinearOperatorLowerTriangular(
                                                                             dim_cov_chols))
            dim_t_student_mixture = tpd.MixtureSameFamily(mixture_distribution=cat_distr,
                                                          components_distribution=dim_t_student_distr)
            dims_mixture_distributions.append(dim_t_student_mixture)

        return dims_mixture_distributions
