import unittest

import tensorflow as tf

from src.models.vaes import vae_model


class StandardVaeTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = (16, 64, 64, 3)
        cls.latent_dim = 32

        cls.input_images = tf.random.normal(shape=cls.input_shape, dtype=tf.float32)
        cls.standard_vae = vae_model.StandardVae(cls.input_shape, cls.latent_dim)

    def test_encode_from_input(self):
        out = self.standard_vae.encode_from_input(self.input_images)
        assert out.shape.as_list() == [self.input_shape[0], 2 * self.latent_dim]

    def test_decode_from_z(self):
        enc_out = self.standard_vae.encode_from_input(self.input_images)
        z = self.standard_vae.reparameterize(enc_out)
        dec_out = self.standard_vae.decode_from_z(z)

        assert z.shape.as_list() == [self.input_shape[0], self.latent_dim]
        assert dec_out.shape.as_list() == list(self.input_shape)

    def test_recon_loss(self):
        dec_out = self.standard_vae.decode_out(self.input_images)
        assert self.standard_vae.recon_loss(self.input_images, dec_out).shape.as_list() == []

    def test_model_variables(self):
        variables = self.standard_vae.model_variables()
        assert len(variables) > 0
        assert all(isinstance(v, tf.Variable) for v in variables)

    def test_delta_vae_is_intercepting(self):
        standard_vae_with_delta = vae_model.StandardVae(self.input_shape, self.latent_dim, delta_val=0.08)
        out = standard_vae_with_delta.encode_from_input(self.input_images)
        assert out.shape.as_list() == [self.input_shape[0], 2 * self.latent_dim]


if __name__ == '__main__':
    unittest.main()
