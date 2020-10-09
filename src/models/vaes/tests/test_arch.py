import unittest

import tensorflow as tf

from src.models.vaes import arch


class ArchTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = (16, 64, 64, 3)
        cls.input_shape128 = (16, 128, 128, 3)
        cls.input_shape84 = (16, 84, 84, 3)
        cls.input_shape96 = (16, 96, 96, 3)

        cls.latent_dim = 32

        cls.input_images = tf.random.normal(shape=cls.input_shape, dtype=tf.float32)
        cls.input_images128 = tf.random.normal(shape=cls.input_shape128, dtype=tf.float32)
        cls.input_images84 = tf.random.normal(shape=cls.input_shape84, dtype=tf.float32)
        cls.input_images96 = tf.random.normal(shape=cls.input_shape96, dtype=tf.float32)

    def test_bigger_arch(self):
        enc_bigger, dec_bigger = arch.create_bigger_arch_func(self.input_shape, self.latent_dim, True)

        enc_out = enc_bigger(self.input_images)
        dec_out = dec_bigger(enc_out[:, :self.latent_dim])

        assert enc_out.shape.as_list() == [self.input_shape[0], 2*self.latent_dim]
        assert dec_out.shape.as_list() == list(self.input_shape)

    def test_bigger_arch128(self):
        enc_bigger, dec_bigger = arch.create_bigger_arch_func(self.input_shape128, self.latent_dim, True)

        enc_out = enc_bigger(self.input_images128)
        dec_out = dec_bigger(enc_out[:, :self.latent_dim])

        assert enc_out.shape.as_list() == [self.input_shape[0], 2*self.latent_dim]
        assert dec_out.shape.as_list() == list(self.input_shape128)

    def test_bigger_arch84(self):
        enc_bigger, dec_bigger = arch.create_bigger_arch_func(self.input_shape84, self.latent_dim, True)

        enc_out = enc_bigger(self.input_images84)
        dec_out = dec_bigger(enc_out[:, :self.latent_dim])

        assert enc_out.shape.as_list() == [self.input_shape[0], 2 * self.latent_dim]
        assert dec_out.shape.as_list() == list(self.input_shape84)

    def test_bigger_arch96(self):
        enc_bigger, dec_bigger = arch.create_bigger_arch_func(self.input_shape96, self.latent_dim, True)

        enc_out = enc_bigger(self.input_images96)
        dec_out = dec_bigger(enc_out[:, :self.latent_dim])

        assert enc_out.shape.as_list() == [self.input_shape[0], 2 * self.latent_dim]
        assert dec_out.shape.as_list() == list(self.input_shape96)


if __name__ == '__main__':
    unittest.main()
