import tensorflow as tf

from utils import logger
from src.models.vaes import latent_space_sampler
from src.models.vaes import conditional_latent_space_sampler


@logger.log
class BhattacharyyaDistCalculator(latent_space_sampler.LatentSpaceSampler):

    def compute_bhattacharyya_dist_for_clusters(self):
        result = {}
        cluster_cov_chols = self.trace_obj['cluster_params']['cov_chol']
        total_count = 0
        for cluster_index, cluster_count in self.cluster_counts.items():
            cluster_bhattacharyya_dist = self.__bhattacharyya_distance(cluster_cov_chols[cluster_index], cluster_count,
                                                                       self.nu_0)
            result[cluster_index] = {'dist': float(cluster_bhattacharyya_dist.numpy()), 'count': cluster_count}
            self.logger.info("Computed bhattacharyya distance for cluster: %d/%d" % (cluster_index,
                                                                                     len(self.cluster_counts)))
            total_count += cluster_count

        result['weighted_dist'] = sum([(float(x['count']) / float(total_count)) * x['dist'] for x in result.values()])
        return result

    @tf.function
    def __bhattacharyya_distance(self, chol_cov, cluster_count, nu_0):
        # chol_cov1 and chol_cov2 are lower triangular matrices - representing covariances of NIW posterior distribution
        cov = tf.linalg.matmul(chol_cov, tf.transpose(chol_cov))
        data_dim = cov.get_shape().as_list()[1]
        div_factor = tf.constant(nu_0, dtype=tf.float32) + tf.cast(cluster_count, dtype=tf.float32) - tf.constant(
            data_dim + 1., dtype=tf.float32)

        iw_cov_mean = cov / div_factor
        iw_cov_diag = tf.linalg.tensor_diag(tf.linalg.diag_part(iw_cov_mean))

        cov_sum = (iw_cov_mean + iw_cov_diag) / tf.constant(2., dtype=tf.float32)

        num = tf.linalg.logdet(cov_sum)
        denom = 0.5 * (tf.linalg.logdet(iw_cov_mean) + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(iw_cov_diag))))

        return 0.5 * (num - denom)


@logger.log
class DKLCalculator(conditional_latent_space_sampler.ConditionalLatentSpaceSampler):

    def __init__(self, samples_num, trace_pkl_path):
        super().__init__(trace_pkl_path)
        self.samples_num = samples_num

        self.joint_distr = self.prepare_t_student_mixture()
        self.marginals_prod = self.prepare_t_student_marginals_prod()

    def calculate_symmetric_dkl(self):
        joint_samples = tf.cast(self.joint_distr.sample(sample_shape=(self.samples_num,)), dtype=tf.float32)
        marginal_prod_samples = [m.sample(sample_shape=(self.samples_num,)) for m in self.marginals_prod]
        marginal_prod_samples = tf.concat(marginal_prod_samples, axis=-1)

        pq_ratio = self.pq_ratio_for_samples(joint_samples)
        qp_ratio = 1./self.pq_ratio_for_samples(marginal_prod_samples)

        return tf.reduce_mean(pq_ratio + qp_ratio).numpy()

    def calculate_joint_and_prod_dkl(self):
        joint_samples = tf.cast(self.joint_distr.sample(sample_shape=(self.samples_num,)), dtype=tf.float32)
        return tf.reduce_mean(self.pq_ratio_for_samples(joint_samples)).numpy()

    @tf.function
    def pq_ratio_for_samples(self, samples_tensor):
        p_log_probs = self.joint_distr.log_prob(samples_tensor)
        prod_log_probs = []
        for i, i_var_distr in enumerate(self.marginals_prod):
            i_var_log_probs = i_var_distr.log_prob(tf.expand_dims(samples_tensor[:, i], axis=-1))
            prod_log_probs.append(i_var_log_probs)

        prod_log_probs = tf.stack(prod_log_probs, axis=-1)
        q_log_probs = tf.reduce_sum(prod_log_probs, axis=-1)

        return p_log_probs - q_log_probs

