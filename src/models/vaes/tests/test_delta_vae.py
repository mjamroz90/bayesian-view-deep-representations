import unittest

import tensorflow as tf

from src.models.vaes import delta_vae


class DeltaVaeTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.delta_vae_obj = delta_vae.DeltaVae(delta_val=0.08)
        cls.sess = tf.Session()

    def test_find_cov_intervals(self):
        for delta_val in [0.5, 0.14, 0.08]:
            delta_val_obj = delta_vae.DeltaVae(delta_val=delta_val)
            left_root, right_root = delta_val_obj.find_cov_intervals()
            assert left_root < right_root
            assert left_root < 1. and right_root > 1.

    def test_reparameterize_to_save_constraint(self):
        dim = 20
        # Prepare mean and cov which are very close to distribution: N(0,1)
        # Mean is slightly shifted by 0.1, covariance is the same - ones vector
        mean = tf.zeros(shape=(1, dim), dtype=tf.float32) + 0.05
        log_cov = tf.math.log(tf.ones(shape=(1, dim), dtype=tf.float32))

        curr_kl_loss = self.sess.run(self.__kl_loss(mean, log_cov))[0]
        assert curr_kl_loss < self.delta_vae_obj.delta_val

        new_mean, new_log_cov = self.delta_vae_obj.reparameterize_to_save_constraint(mean, log_cov)
        new_kl_loss = self.sess.run(self.__kl_loss(new_mean, new_log_cov))[0]

        assert new_kl_loss > self.delta_vae_obj.delta_val

    @staticmethod
    def __kl_loss(mean, log_cov):
        kl_loss = -0.5 * tf.reduce_sum(1 + log_cov - tf.square(mean) - tf.exp(log_cov), axis=1)
        return kl_loss


if __name__ == '__main__':
    unittest.main()
