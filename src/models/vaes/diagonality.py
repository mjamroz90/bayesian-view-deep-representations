import tensorflow as tf

from utils import logger
from src.models.vaes import conditional_latent_space_sampler


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

